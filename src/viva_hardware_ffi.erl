%% VIVA Hardware FFI - Serial Port Communication
%% Stub implementation - returns error for now

-module(viva_hardware_ffi).
-export([open_serial_port/2]).

%% Open serial port - stub that returns error
%% Real implementation needs erlang-serial or similar
open_serial_port(_Device, _Baud) ->
    {error, not_implemented}.
