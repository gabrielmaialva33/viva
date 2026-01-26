%% VIVA Serial FFI - Pure Erlang serial communication
%% No Python, no external dependencies
-module(viva_serial_ffi).
-export([configure_port/1, open_device/1, write_port/2, read_port/1, read_line_port/2, close_port/1]).

%% Configure serial port using stty
%% Config is a Gleam record: {config, Device, Baud, DataBits, StopBits, Parity}
configure_port({config, DeviceBin, Baud, _DataBits, _StopBits, _Parity}) ->
    Device = binary_to_list(DeviceBin),

    %% Build stty command
    Cmd = io_lib:format(
        "stty -F ~s ~p raw -echo -echoe -echok -echoctl -echoke",
        [Device, Baud]
    ),

    Result = os:cmd(lists:flatten(Cmd)),
    %% stty returns empty string on success
    Result == [].

%% Open device as Erlang port
-spec open_device(binary()) -> port().
open_device(DeviceBin) ->
    Device = binary_to_list(DeviceBin),

    %% Open the device file directly
    case file:open(Device, [read, write, binary, raw]) of
        {ok, Fd} ->
            %% Store fd in process dictionary for now
            %% Return a pseudo-port structure
            {serial_fd, Fd, Device};
        {error, Reason} ->
            error({open_failed, Reason})
    end.

%% Write to serial port
-spec write_port({serial_fd, file:fd(), string()}, binary()) -> boolean().
write_port({serial_fd, Fd, _Device}, Data) ->
    case file:write(Fd, Data) of
        ok -> true;
        {error, _} -> false
    end.

%% Read from serial port (non-blocking)
-spec read_port({serial_fd, file:fd(), string()}) -> {some, binary()} | none.
read_port({serial_fd, Fd, _Device}) ->
    case file:read(Fd, 256) of
        {ok, Data} -> {some, Data};
        eof -> none;
        {error, _} -> none
    end.

%% Read line with timeout
-spec read_line_port({serial_fd, file:fd(), string()}, integer()) -> {some, binary()} | none.
read_line_port({serial_fd, Fd, _Device}, TimeoutMs) ->
    read_until_newline(Fd, <<>>, TimeoutMs).

read_until_newline(Fd, Acc, TimeoutMs) when TimeoutMs > 0 ->
    case file:read(Fd, 1) of
        {ok, <<"\n">>} ->
            {some, Acc};
        {ok, <<C>>} ->
            read_until_newline(Fd, <<Acc/binary, C>>, TimeoutMs);
        eof ->
            timer:sleep(10),
            read_until_newline(Fd, Acc, TimeoutMs - 10);
        {error, _} ->
            timer:sleep(10),
            read_until_newline(Fd, Acc, TimeoutMs - 10)
    end;
read_until_newline(_Fd, Acc, _TimeoutMs) ->
    case Acc of
        <<>> -> none;
        _ -> {some, Acc}
    end.

%% Close port
-spec close_port({serial_fd, file:fd(), string()}) -> ok.
close_port({serial_fd, Fd, _Device}) ->
    file:close(Fd),
    ok.
