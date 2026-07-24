// =====================================================================
// JOLT ⇆ IREE runtime shim.
//
// A thin C adapter over IREE's runtime C API, linked with iree::runtime (no
// LLVM) into libjolt_iree, which Julia dlopens and ccalls. It runs a .vmfb
// in-process: create a session (cached), push the shared `vars` arena + typed
// activation inputs, invoke, pop each output.
//
// vars (weights + any assigned variables) travel as ONE flat, READ-ONLY call
// argument — the `vars` ComponentArray's page-aligned arena — imported ZERO-COPY
// as a host allocation (jolt_push_arena): the compiled graph slices each variable
// out of it, so a Julia-side mutation between calls (an optimiser step, a mode
// flip) is seen by the next invoke with no copy. `assign!` updates come back as
// ordinary outputs that Julia copies into the arena's slots — the graph never
// writes the arena, so no aliasing is involved.
//
// Zero-copy import works on unified memory (CPU heap, Apple Metal shared). On a
// backend that can't alias a host allocation (a discrete GPU), or for an arena
// that isn't import-eligible, jolt_push_arena falls back to copy-in. The
// per-session capability is decided once by jolt_import_probe.
//
// Activation inputs also copy in (jolt_push_input). JOLT reconciles Julia
// column-major vs IREE row-major with the reverse-dims graph convention, so the
// host boundary is transpose-free.
// =====================================================================
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "iree/runtime/api.h"

typedef struct {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
} jolt_session;

typedef struct {
  iree_runtime_call_t call;
  jolt_session* s;
  iree_hal_buffer_view_t* pending;   // most-recently popped output, awaiting read
} jolt_call;

void jolt_session_release(jolt_session* s);   // fwd decl (create's error path)

static void logst(const char* where, iree_status_t st) {
  char buf[2048]; iree_host_size_t len = 0;
  if (iree_status_format(st, sizeof(buf), buf, &len))
    fprintf(stderr, "[jolt %s] %.*s\n", where, (int)len, buf);
  iree_status_ignore(st);
}

// element-type codes shared with Julia: 0 f32,1 f64,2 f16,3 i32,4 i64,5 i8,6 bool
static iree_hal_element_type_t ecode2hal(int c) {
  switch (c) {
    case 0: return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case 1: return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    case 2: return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case 3: return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case 4: return IREE_HAL_ELEMENT_TYPE_SINT_64;
    case 5: return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case 6: return IREE_HAL_ELEMENT_TYPE_BOOL_8;
    default: return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  }
}
static int hal2ecode(iree_hal_element_type_t t) {
  switch (t) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: return 0;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64: return 1;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16: return 2;
    // StableHLO integers are SIGNLESS, so outputs come back as INT_* not SINT_*;
    // map both (same width) to the same Julia integer type.
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_ELEMENT_TYPE_INT_32:   return 3;
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_ELEMENT_TYPE_INT_64:   return 4;
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
    case IREE_HAL_ELEMENT_TYPE_INT_8:    return 5;
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:   return 6;
    default: return -1;
  }
}

// Import the caller's arena host allocation as a non-owning, READ-ONLY HAL buffer.
// On CPU heap / Metal shared this wraps the pointer with NO copy (dispatches read
// it directly); on a backend that can't alias host memory as device-local the
// import fails and the caller copies in. Requesting DEVICE_LOCAL is what makes a
// discrete GPU reject the import (its host allocations aren't device-local), which
// is the signal to fall back. The null release callback means IREE never frees it
// — the memory belongs to Julia and MUST outlive the buffer.
static iree_status_t jolt_import_arena(jolt_session* s, void* ptr, int64_t nbytes,
                                       iree_hal_buffer_t** out_buffer) {
  iree_hal_external_buffer_t ext;
  memset(&ext, 0, sizeof(ext));
  ext.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  ext.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
  ext.size = (iree_device_size_t)nbytes;
  ext.handle.host_allocation.ptr = ptr;
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type  = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_READ;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_import_buffer(
      iree_runtime_session_device_allocator(s->session), params, &ext,
      iree_hal_buffer_release_callback_null(), out_buffer);
}

// Create a session over `vmfb`, using the named HAL `driver` ("local-task" for
// CPU, "metal" for the Mac GPU). vars are NOT bound here — they arrive per call
// as jolt_push_arena's input 0.
jolt_session* jolt_session_create(const char* vmfb_path, const char* driver) {
  jolt_session* s = (jolt_session*)calloc(1, sizeof(jolt_session));
  if (!s) return NULL;
  iree_runtime_instance_options_t opts;
  iree_runtime_instance_options_initialize(&opts);
  iree_runtime_instance_options_use_all_available_drivers(&opts);
  if (!iree_status_is_ok(iree_runtime_instance_create(&opts, iree_allocator_system(), &s->instance))) goto err;
  if (!iree_status_is_ok(iree_runtime_instance_try_create_default_device(
        s->instance, iree_make_cstring_view(driver), &s->device))) goto err;
  iree_runtime_session_options_t sopts;
  iree_runtime_session_options_initialize(&sopts);
  if (!iree_status_is_ok(iree_runtime_session_create_with_device(
        s->instance, &sopts, s->device, iree_runtime_instance_host_allocator(s->instance), &s->session))) goto err;
  { iree_status_t st = iree_runtime_session_append_bytecode_module_from_file(s->session, vmfb_path);
    if (!iree_status_is_ok(st)) { logst("load", st); goto err; } }
  return s;
err:
  jolt_session_release(s);
  return NULL;
}

void jolt_session_release(jolt_session* s) {
  if (!s) return;
  if (s->session)  iree_runtime_session_release(s->session);
  if (s->device)   iree_hal_device_release(s->device);
  if (s->instance) iree_runtime_instance_release(s->instance);
  free(s);
}

// Can this device import the caller's arena zero-copy for reads? Attempt the
// host-allocation import, map it, and require the mapped pointer to be IDENTICAL
// to the caller's — the only reliable signal (IMPORTABLE is unconditional on the
// CPU heap and a false-negative on Metal). Returns 1 if zero-copy, else 0 (⇒
// copy-in per call).
int jolt_import_probe(jolt_session* s, void* ptr, int64_t nbytes) {
  if (!s || !ptr || nbytes <= 0) return 0;
  iree_hal_buffer_t* buf = NULL;
  iree_status_t st = jolt_import_arena(s, ptr, nbytes, &buf);
  if (!iree_status_is_ok(st)) { iree_status_ignore(st); return 0; }
  iree_hal_buffer_mapping_t m; memset(&m, 0, sizeof(m));
  st = iree_hal_buffer_map_range(buf, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ, 0, (iree_device_size_t)nbytes, &m);
  int ok = iree_status_is_ok(st) && (m.contents.data == (uint8_t*)ptr);
  if (iree_status_is_ok(st)) iree_hal_buffer_unmap_range(&m);
  else iree_status_ignore(st);
  iree_hal_buffer_release(buf);
  return ok ? 1 : 0;
}

jolt_call* jolt_call_begin(jolt_session* s, const char* func) {
  jolt_call* c = (jolt_call*)calloc(1, sizeof(jolt_call));
  if (!c) return NULL;
  c->s = s;
  iree_status_t st = iree_runtime_call_initialize_by_name(s->session, iree_make_cstring_view(func), &c->call);
  if (!iree_status_is_ok(st)) { logst("call_init", st); free(c); return NULL; }
  return c;
}

// Push the `vars` arena as READ-ONLY call input 0. If `try_zerocopy`, attempt the
// non-owning host-allocation import (wrapping Julia's bytes directly); on success
// push that view and return 1. If the import is disabled or fails (unsupported
// backend, non-import-eligible pointer), copy the arena into a device buffer and
// return 0. Returns <0 on hard error. When 1, the arena MUST stay live
// (GC.@preserve) through jolt_call_end.
int jolt_push_arena(jolt_call* c, int ecode, int rank, const int64_t* dims,
                    void* ptr, int64_t nbytes, int try_zerocopy) {
  iree_hal_dim_t shape[16];
  for (int i = 0; i < rank; i++) shape[i] = (iree_hal_dim_t)dims[i];
  iree_hal_element_type_t et = ecode2hal(ecode);
  iree_hal_buffer_view_t* view = NULL;
  int mode = -1;

  if (try_zerocopy) {
    iree_hal_buffer_t* buf = NULL;
    iree_status_t st = jolt_import_arena(c->s, ptr, nbytes, &buf);
    if (iree_status_is_ok(st)) {
      st = iree_hal_buffer_view_create(buf, (iree_host_size_t)rank, shape, et,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          iree_runtime_session_host_allocator(c->s->session), &view);
      iree_hal_buffer_release(buf);   // the view retains it
      if (iree_status_is_ok(st)) { mode = 1; }
      else { iree_status_ignore(st); view = NULL; }
    } else { iree_status_ignore(st); }
  }

  if (mode < 0) {   // copy-in fallback (device buffer holds a copy of the arena)
    iree_hal_buffer_params_t params; memset(&params, 0, sizeof(params));
    params.type  = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    iree_status_t st = iree_hal_buffer_view_allocate_buffer_copy(
        iree_runtime_session_device(c->s->session),
        iree_runtime_session_device_allocator(c->s->session),
        (iree_host_size_t)rank, shape, et, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params, iree_make_const_byte_span(ptr, (iree_host_size_t)nbytes), &view);
    if (!iree_status_is_ok(st)) { logst("arena_copy", st); return -1; }
    mode = 0;
  }

  iree_status_t st = iree_runtime_call_inputs_push_back_buffer_view(&c->call, view);
  iree_hal_buffer_view_release(view);
  if (!iree_status_is_ok(st)) { logst("arena_push", st); return -1; }
  return mode;
}

// Push a typed activation input (copied in; host<->device).
int jolt_push_input(jolt_call* c, int ecode, int rank, const int64_t* dims,
                    const void* data, int64_t nbytes) {
  iree_hal_dim_t shape[16];
  for (int i = 0; i < rank; i++) shape[i] = (iree_hal_dim_t)dims[i];
  iree_hal_buffer_params_t params; memset(&params, 0, sizeof(params));
  params.type  = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree_hal_buffer_view_t* view = NULL;
  iree_status_t st = iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(c->s->session),
      iree_runtime_session_device_allocator(c->s->session),
      (iree_host_size_t)rank, shape, ecode2hal(ecode), IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params, iree_make_const_byte_span(data, (iree_host_size_t)nbytes), &view);
  if (!iree_status_is_ok(st)) { logst("alloc_in", st); return 1; }
  st = iree_runtime_call_inputs_push_back_buffer_view(&c->call, view);
  iree_hal_buffer_view_release(view);
  return iree_status_is_ok(st) ? 0 : 2;
}

int jolt_invoke(jolt_call* c) {
  iree_status_t st = iree_runtime_call_invoke(&c->call, 0);
  if (!iree_status_is_ok(st)) { logst("invoke", st); return 1; }
  return 0;
}

// Pop the next output buffer view; fill ecode/rank/dims (dims capacity ≥ 16). 0 ok, 1 none.
int jolt_output_next(jolt_call* c, int* ecode, int* rank, int64_t* dims) {
  if (c->pending) { iree_hal_buffer_view_release(c->pending); c->pending = NULL; }
  iree_status_t st = iree_runtime_call_outputs_pop_front_buffer_view(&c->call, &c->pending);
  if (!iree_status_is_ok(st)) { iree_status_ignore(st); return 1; }
  iree_host_size_t r = iree_hal_buffer_view_shape_rank(c->pending);
  *rank  = (int)r;
  *ecode = hal2ecode(iree_hal_buffer_view_element_type(c->pending));
  for (iree_host_size_t i = 0; i < r; i++) dims[i] = (int64_t)iree_hal_buffer_view_shape_dim(c->pending, i);
  return 0;
}

// Copy the current pending output into dst (nbytes) and release it.
int jolt_output_read(jolt_call* c, void* dst, int64_t nbytes) {
  if (!c->pending) return 1;
  iree_status_t st = iree_hal_device_transfer_d2h(
      iree_runtime_session_device(c->s->session), iree_hal_buffer_view_buffer(c->pending), 0,
      dst, (iree_host_size_t)nbytes, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  iree_hal_buffer_view_release(c->pending); c->pending = NULL;
  if (!iree_status_is_ok(st)) { logst("d2h", st); return 2; }
  return 0;
}

void jolt_call_end(jolt_call* c) {
  if (!c) return;
  if (c->pending) iree_hal_buffer_view_release(c->pending);
  iree_runtime_call_deinitialize(&c->call);
  free(c);
}
