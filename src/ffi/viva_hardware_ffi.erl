%% VIVA Hardware FFI - Serial Port Communication
%% Real implementation using Erlang ports for Arduino/ESP32

-module(viva_hardware_ffi).
-export([open_serial_port/2, configure_serial/2]).

%% Open serial port via Erlang port
%% Device: "/dev/ttyUSB0", "/dev/ttyACM0", etc
%% Baud: 9600, 115200, etc
open_serial_port(Device, Baud) ->
    %% Configure serial port first (Linux/WSL)
    configure_serial(Device, Baud),
    %% Open as spawned port with cat for reading
    Port = open_port({spawn, "cat " ++ binary_to_list(Device)},
                     [binary, {line, 256}, exit_status]),
    Port.

%% Configure serial port settings using stty
configure_serial(Device, Baud) ->
    DeviceStr = binary_to_list(Device),
    BaudStr = integer_to_list(Baud),
    %% Configure: raw mode, baud rate, 8N1
    Cmd = "stty -F " ++ DeviceStr ++ " " ++ BaudStr ++ " cs8 -cstopb -parenb raw -echo 2>/dev/null",
    os:cmd(Cmd),
    ok.
