%% VIVA Performance FFI
%% Erlang helper functions for process performance metrics
-module(viva_perf_ffi).
-export([get_reductions/0, get_heap_size/0, get_gc_count/0, monotonic_time_micro/0]).

%% Get monotonic time in microseconds
monotonic_time_micro() ->
    erlang:monotonic_time(microsecond).

%% Get current process reductions (work units)
get_reductions() ->
    {reductions, R} = erlang:process_info(self(), reductions),
    R.

%% Get current process heap size in words
get_heap_size() ->
    {heap_size, H} = erlang:process_info(self(), heap_size),
    H.

%% Get number of GC collections for current process
get_gc_count() ->
    {garbage_collection, GCInfo} = erlang:process_info(self(), garbage_collection),
    proplists:get_value(minor_gcs, GCInfo, 0).
