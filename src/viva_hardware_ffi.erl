%% VIVA Hardware FFI - Erlang bridge for serial port communication
%%
%% This module provides the FFI functions for the Gleam port_manager.

-module(viva_hardware_ffi).
-export([open_serial_port/2, configure_port/2]).

%% Open a serial port using Python serial bridge
%% The bridge handles bidirectional communication properly.
-spec open_serial_port(binary(), integer()) -> port().
open_serial_port(Device, Baud) ->
    DeviceStr = binary_to_list(Device),
    BaudStr = integer_to_list(Baud),

    %% Find the serial bridge script
    ScriptPath = filename:join([code:priv_dir(viva), "..", "scripts", "serial_bridge.py"]),

    %% Fall back to relative path if priv_dir fails
    ActualPath = case filelib:is_file(ScriptPath) of
        true -> ScriptPath;
        false -> "scripts/serial_bridge.py"
    end,

    %% Open port using Python serial bridge
    Cmd = "python3 " ++ ActualPath ++ " " ++ DeviceStr ++ " " ++ BaudStr,

    Port = open_port({spawn, Cmd}, [
        binary,
        stream,
        use_stdio,
        exit_status
    ]),

    Port.

%% Alternative: configure port settings
-spec configure_port(binary(), map()) -> ok | {error, term()}.
configure_port(Device, Settings) ->
    DeviceStr = binary_to_list(Device),
    Baud = maps:get(baud, Settings, 115200),
    DataBits = maps:get(data_bits, Settings, 8),
    StopBits = maps:get(stop_bits, Settings, 1),
    Parity = maps:get(parity, Settings, none),

    BaudStr = integer_to_list(Baud),
    DataBitsStr = "cs" ++ integer_to_list(DataBits),
    StopBitsStr = case StopBits of
        1 -> "-cstopb";
        2 -> "cstopb"
    end,
    ParityStr = case Parity of
        none -> "-parenb";
        even -> "parenb -parodd";
        odd -> "parenb parodd"
    end,

    Cmd = io_lib:format("stty -F ~s ~s ~s ~s ~s raw -echo",
                        [DeviceStr, BaudStr, DataBitsStr, StopBitsStr, ParityStr]),

    case os:cmd(lists:flatten(Cmd)) of
        [] -> ok;
        Error -> {error, Error}
    end.

%% For a proper implementation, you would use something like:
%%
%% open_serial_port_proper(Device, Baud) ->
%%     %% Using circuits_uart style (Elixir/Nerves)
%%     {ok, Pid} = circuits_uart:start_link(),
%%     ok = circuits_uart:open(Pid, Device, [
%%         {speed, Baud},
%%         {active, true},
%%         {framing, none}  %% We handle COBS framing in Gleam
%%     ]),
%%     Pid.
