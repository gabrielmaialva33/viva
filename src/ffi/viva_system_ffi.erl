%% VIVA System FFI
%% Erlang helper functions for system metrics
-module(viva_system_ffi).
-export([system_time_seconds/0, scheduler_count/0]).

%% Get current Unix timestamp in seconds
system_time_seconds() ->
    erlang:system_time(second).

%% Get number of Erlang schedulers (logical CPUs)
scheduler_count() ->
    erlang:system_info(schedulers).
