%% VIVA Body FFI - Serial communication with Arduino
-module(viva_body_ffi).
-export([open_serial/2, port_write/2, port_close/1, start_reader/2]).

%% Open serial port using stty + cat (pure Unix, no Python)
-spec open_serial(binary(), integer()) -> port() | undefined.
open_serial(Device, Baud) ->
    DeviceStr = binary_to_list(Device),
    BaudStr = integer_to_list(Baud),

    %% Configure serial port with stty
    SttyCmd = io_lib:format("stty -F ~s ~s raw -echo", [DeviceStr, BaudStr]),
    os:cmd(lists:flatten(SttyCmd)),

    %% Wait for Arduino reset
    timer:sleep(2000),

    %% Open device directly as Erlang port
    Port = open_port({spawn, "cat " ++ DeviceStr}, [
        binary,
        stream,
        {line, 256}
    ]),
    Port.

%% Escreve na porta
-spec port_write(port(), binary()) -> boolean().
port_write(Port, Data) ->
    try
        port_command(Port, Data),
        true
    catch
        _:_ -> false
    end.

%% Fecha porta
-spec port_close(port()) -> boolean().
port_close(Port) ->
    try
        erlang:port_close(Port),
        true
    catch
        _:_ -> false
    end.

%% Inicia reader process que envia mensagens para um Pid
-spec start_reader(port(), pid()) -> pid().
start_reader(Port, TargetPid) ->
    spawn(fun() -> reader_loop(Port, TargetPid) end).

reader_loop(Port, TargetPid) ->
    receive
        {Port, {data, {eol, Line}}} ->
            TargetPid ! {serial_line, Line},
            reader_loop(Port, TargetPid);
        {Port, {data, {noeol, _}}} ->
            reader_loop(Port, TargetPid);
        {Port, {exit_status, _}} ->
            TargetPid ! serial_closed;
        stop ->
            ok
    end.
