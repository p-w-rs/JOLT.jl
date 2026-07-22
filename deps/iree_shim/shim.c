// Myelin IREE runtime shim: a thin C adapter over IREE's runtime C API, compiled together
// with iree::runtime (no LLVM) into libmyelin_iree. Julia ccalls these small functions to
// run a .vmfb in-process: create a session once (cache it), push typed inputs, invoke, then
// pop each output (querying its runtime shape, which may come from dynamic dims).
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "iree/runtime/api.h"

typedef struct {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
} ms_session;

typedef struct {
  iree_runtime_call_t call;
  ms_session* s;
  iree_hal_buffer_view_t* pending;   // most-recently popped output, awaiting read
} ms_call;

void myelin_session_release(ms_session* s);   // fwd decl (used in create's error path)

static void logst(const char* where, iree_status_t st) {
  char buf[2048]; iree_host_size_t len = 0;
  if (iree_status_format(st, sizeof(buf), buf, &len))
    fprintf(stderr, "[myelin %s] %.*s\n", where, (int)len, buf);
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
    case IREE_HAL_ELEMENT_TYPE_SINT_32:  return 3;
    case IREE_HAL_ELEMENT_TYPE_SINT_64:  return 4;
    case IREE_HAL_ELEMENT_TYPE_SINT_8:   return 5;
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:   return 6;
    default: return -1;
  }
}

ms_session* myelin_session_create(const char* vmfb_path) {
  ms_session* s = (ms_session*)calloc(1, sizeof(ms_session));
  if (!s) return NULL;
  iree_runtime_instance_options_t opts;
  iree_runtime_instance_options_initialize(&opts);
  iree_runtime_instance_options_use_all_available_drivers(&opts);
  if (!iree_status_is_ok(iree_runtime_instance_create(&opts, iree_allocator_system(), &s->instance))) goto err;
  if (!iree_status_is_ok(iree_runtime_instance_try_create_default_device(
        s->instance, iree_make_cstring_view("local-task"), &s->device))) goto err;
  iree_runtime_session_options_t sopts;
  iree_runtime_session_options_initialize(&sopts);
  if (!iree_status_is_ok(iree_runtime_session_create_with_device(
        s->instance, &sopts, s->device, iree_runtime_instance_host_allocator(s->instance), &s->session))) goto err;
  { iree_status_t st = iree_runtime_session_append_bytecode_module_from_file(s->session, vmfb_path);
    if (!iree_status_is_ok(st)) { logst("load", st); goto err; } }
  return s;
err:
  myelin_session_release(s);
  return NULL;
}

void myelin_session_release(ms_session* s) {
  if (!s) return;
  if (s->session)  iree_runtime_session_release(s->session);
  if (s->device)   iree_hal_device_release(s->device);
  if (s->instance) iree_runtime_instance_release(s->instance);
  free(s);
}

ms_call* myelin_call_begin(ms_session* s, const char* func) {
  ms_call* c = (ms_call*)calloc(1, sizeof(ms_call));
  if (!c) return NULL;
  c->s = s;
  iree_status_t st = iree_runtime_call_initialize_by_name(s->session, iree_make_cstring_view(func), &c->call);
  if (!iree_status_is_ok(st)) { logst("call_init", st); free(c); return NULL; }
  return c;
}

int myelin_push_input(ms_call* c, int ecode, int rank, const int64_t* dims,
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

int myelin_invoke(ms_call* c) {
  iree_status_t st = iree_runtime_call_invoke(&c->call, 0);
  if (!iree_status_is_ok(st)) { logst("invoke", st); return 1; }
  return 0;
}

// Pop the next output buffer view; fill ecode/rank/dims (dims capacity ≥16). 0 ok, 1 none.
int myelin_output_next(ms_call* c, int* ecode, int* rank, int64_t* dims) {
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
int myelin_output_read(ms_call* c, void* dst, int64_t nbytes) {
  if (!c->pending) return 1;
  iree_status_t st = iree_hal_device_transfer_d2h(
      iree_runtime_session_device(c->s->session), iree_hal_buffer_view_buffer(c->pending), 0,
      dst, (iree_host_size_t)nbytes, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  iree_hal_buffer_view_release(c->pending); c->pending = NULL;
  if (!iree_status_is_ok(st)) { logst("d2h", st); return 2; }
  return 0;
}

void myelin_call_end(ms_call* c) {
  if (!c) return;
  if (c->pending) iree_hal_buffer_view_release(c->pending);
  iree_runtime_call_deinitialize(&c->call);
  free(c);
}
