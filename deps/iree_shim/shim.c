// =====================================================================
// JOLT ⇆ IREE runtime shim.
//
// A thin C adapter over IREE's runtime C API, linked with iree::runtime (no
// LLVM) into libjolt_iree, which Julia dlopens and ccalls. It runs a .vmfb
// in-process: create a session (cached), push typed inputs, invoke, pop each
// output (querying its runtime shape, which may be dynamic).
//
// This is the CORRECTNESS-phase shim: inputs are copied in and outputs copied
// out (host<->device), and JOLT reconciles Julia column-major vs IREE row-major
// by transposing at the boundary. The zero-copy phase replaces the input path
// with iree_hal_allocator_import_buffer (wrap Julia's aligned arena directly).
// =====================================================================
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "iree/runtime/api.h"
// Zero-copy weights (io_parameters): the program's :variables are served by a
// runtime parameter provider that ALIASES Julia's host arena instead of being
// pushed as call inputs. See jolt_session_create.
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/io/parameters/module.h"

typedef struct {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
  iree_io_parameter_index_t* param_index;    // owns the key→host-range mapping
  iree_io_parameter_provider_t* provider;     // serves it to io_parameters.load
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

// Create a session over `vmfb`, using the named HAL `driver` (e.g. "local-task"
// for CPU, "metal" for the Mac GPU).
//
// If `param_nbytes > 0`, wire up zero-copy weights: build a parameter index with
// ONE entry (`scope`::`key` → the caller's host range [param_data, param_nbytes))
// backed by a non-owning host-allocation file handle, wrap it in a provider, and
// append the io_parameters module BEFORE the program module so the program's
// `io_parameters.load` resolves against it. On the local/CPU backend the load
// imports the host pointer with no copy, so the program reads Julia's arena in
// place. `param_data` MUST stay live (GC.@preserve) and 64-byte aligned for the
// lifetime of the session. `param_nbytes == 0` ⇒ no parameters (e.g. a graph
// with no variables), and the session is built exactly as before.
jolt_session* jolt_session_create(const char* vmfb_path, const char* driver,
                                  const char* scope, const char* key,
                                  void* param_data, int64_t param_nbytes) {
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

  if (param_nbytes > 0) {
    iree_allocator_t host = iree_runtime_instance_host_allocator(s->instance);
    iree_vm_instance_t* vm = iree_runtime_instance_vm_instance(s->instance);
    // Non-owning wrap of Julia's arena (READ: IREE only reads weights; Julia
    // writes them in place between calls). The null release callback means we
    // keep ownership — the memory must outlive the session.
    iree_io_file_handle_t* handle = NULL;
    if (!iree_status_is_ok(iree_io_file_handle_wrap_host_allocation(
            IREE_IO_FILE_ACCESS_READ,
            iree_make_byte_span(param_data, (iree_host_size_t)param_nbytes),
            iree_io_file_handle_release_callback_null(), host, &handle))) { logst("wrap_host", iree_ok_status()); goto err; }
    iree_status_t st = iree_io_parameter_index_create(host, &s->param_index);
    if (iree_status_is_ok(st)) {
      iree_io_parameter_index_entry_t entry;
      memset(&entry, 0, sizeof(entry));
      entry.key    = iree_make_cstring_view(key);
      entry.length = (uint64_t)param_nbytes;
      entry.type   = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
      entry.storage.file.handle = handle;
      entry.storage.file.offset = 0;
      st = iree_io_parameter_index_add(s->param_index, &entry);
    }
    iree_io_file_handle_release(handle);   // the index retained it
    if (!iree_status_is_ok(st)) { logst("param_index", st); goto err; }
    if (!iree_status_is_ok(iree_io_parameter_index_provider_create(
            iree_make_cstring_view(scope), s->param_index,
            IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
            host, &s->provider))) { logst("provider", iree_ok_status()); goto err; }
    iree_vm_module_t* pmod = NULL;
    if (!iree_status_is_ok(iree_io_parameters_module_create(vm, 1, &s->provider, host, &pmod))) {
      logst("params_module", iree_ok_status()); goto err; }
    iree_status_t sa = iree_runtime_session_append_module(s->session, pmod);
    iree_vm_module_release(pmod);          // the context retained it
    if (!iree_status_is_ok(sa)) { logst("append_params", sa); goto err; }
  }

  { iree_status_t st = iree_runtime_session_append_bytecode_module_from_file(s->session, vmfb_path);
    if (!iree_status_is_ok(st)) { logst("load", st); goto err; } }
  return s;
err:
  jolt_session_release(s);
  return NULL;
}

void jolt_session_release(jolt_session* s) {
  if (!s) return;
  if (s->session)     iree_runtime_session_release(s->session);   // drops the module's provider ref
  if (s->provider)    iree_io_parameter_provider_release(s->provider);
  if (s->param_index) iree_io_parameter_index_release(s->param_index);
  if (s->device)      iree_hal_device_release(s->device);
  if (s->instance)    iree_runtime_instance_release(s->instance);
  free(s);
}

jolt_call* jolt_call_begin(jolt_session* s, const char* func) {
  jolt_call* c = (jolt_call*)calloc(1, sizeof(jolt_call));
  if (!c) return NULL;
  c->s = s;
  iree_status_t st = iree_runtime_call_initialize_by_name(s->session, iree_make_cstring_view(func), &c->call);
  if (!iree_status_is_ok(st)) { logst("call_init", st); free(c); return NULL; }
  return c;
}

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
