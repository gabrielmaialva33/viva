%% viva_vision_ffi.erl - Vision processing FFI
%% Integrates with NV-CLIP (localhost:8050) and PaddleOCR (localhost:8020)
-module(viva_vision_ffi).
-export([
    read_file/1,
    clip_embed/2,
    ocr_extract/2,
    tensor_to_list/1
]).

%% Read binary file
-spec read_file(binary()) -> {ok, binary()} | {error, binary()}.
read_file(PathBin) ->
    Path = binary_to_list(PathBin),
    case file:read_file(Path) of
        {ok, Data} -> {ok, Data};
        {error, Reason} -> {error, atom_to_binary(Reason, utf8)}
    end.

%% Get CLIP embedding from NV-CLIP server
%% POST image to localhost:8050/embed, returns 1024-dim float list
-spec clip_embed(binary(), binary()) -> {ok, [float()]} | {error, binary()}.
clip_embed(ImageBytes, UrlBin) ->
    Url = binary_to_list(UrlBin) ++ "/embed",
    case http_post_binary(Url, ImageBytes, "application/octet-stream") of
        {ok, ResponseBody} ->
            case parse_embedding_json(ResponseBody) of
                {ok, Embedding} -> {ok, Embedding};
                {error, Reason} -> {error, Reason}
            end;
        {error, Reason} ->
            {error, Reason}
    end.

%% Extract text using PaddleOCR
%% POST image to localhost:8020/ocr, returns text string
-spec ocr_extract(binary(), binary()) -> {ok, binary()} | {error, binary()}.
ocr_extract(ImageBytes, UrlBin) ->
    Url = binary_to_list(UrlBin) ++ "/ocr",
    case http_post_binary(Url, ImageBytes, "application/octet-stream") of
        {ok, ResponseBody} ->
            case parse_ocr_json(ResponseBody) of
                {ok, Text} -> {ok, Text};
                {error, Reason} -> {error, Reason}
            end;
        {error, Reason} ->
            {error, Reason}
    end.

%% Convert tensor (any term with data field) to list
-spec tensor_to_list(term()) -> [float()].
tensor_to_list(Tensor) when is_map(Tensor) ->
    case maps:get(data, Tensor, undefined) of
        undefined -> [];
        Data when is_list(Data) -> Data;
        _ -> []
    end;
tensor_to_list(Tensor) when is_tuple(Tensor) ->
    %% Handle Gleam record: {tensor, Data, Shape} or {hrr, Vector, Dim}
    case Tensor of
        {tensor, Data, _Shape} when is_list(Data) -> Data;
        {hrr, Vector, _Dim} -> tensor_to_list(Vector);
        {'Tensor', Data, _Shape} when is_list(Data) -> Data;
        {'HRR', Vector, _Dim} -> tensor_to_list(Vector);
        _ ->
            %% Try to extract from tuple elements
            TupleList = tuple_to_list(Tensor),
            case lists:filter(fun is_list/1, TupleList) of
                [L | _] -> L;
                [] -> []
            end
    end;
tensor_to_list(List) when is_list(List) ->
    List;
tensor_to_list(_) ->
    [].

%% ============================================================================
%% HTTP Client (using httpc from inets)
%% ============================================================================

http_post_binary(Url, Body, ContentType) ->
    %% Ensure inets is started
    case application:ensure_all_started(inets) of
        {ok, _} -> ok;
        {error, {already_started, _}} -> ok
    end,
    case application:ensure_all_started(ssl) of
        {ok, _} -> ok;
        {error, {already_started, _}} -> ok
    end,

    Request = {Url, [], ContentType, Body},
    HTTPOptions = [{timeout, 10000}, {connect_timeout, 5000}],
    Options = [{body_format, binary}],

    case httpc:request(post, Request, HTTPOptions, Options) of
        {ok, {{_, 200, _}, _Headers, ResponseBody}} ->
            {ok, ResponseBody};
        {ok, {{_, StatusCode, _}, _Headers, ResponseBody}} ->
            {error, iolist_to_binary([
                <<"HTTP ">>, integer_to_binary(StatusCode),
                <<": ">>, ResponseBody
            ])};
        {error, {failed_connect, _}} ->
            {error, <<"Connection failed - is the service running?">>};
        {error, timeout} ->
            {error, <<"Request timeout">>};
        {error, Reason} ->
            {error, iolist_to_binary(io_lib:format("~p", [Reason]))}
    end.

%% ============================================================================
%% JSON Parsing (simple, no external deps)
%% ============================================================================

%% Parse embedding JSON response: {"embedding": [0.1, 0.2, ...]}
parse_embedding_json(JsonBin) ->
    %% Simple parsing - look for array of numbers
    case extract_json_array(JsonBin) of
        {ok, Numbers} -> {ok, Numbers};
        error ->
            %% Try alternative format: just a flat array [0.1, 0.2, ...]
            case parse_flat_array(JsonBin) of
                {ok, Numbers} -> {ok, Numbers};
                error -> {error, <<"Failed to parse embedding JSON">>}
            end
    end.

%% Parse OCR JSON response: {"text": "extracted text"} or {"results": [...]}
parse_ocr_json(JsonBin) ->
    case extract_json_string(JsonBin, <<"text">>) of
        {ok, Text} -> {ok, Text};
        error ->
            %% Try results array format
            case extract_json_string(JsonBin, <<"result">>) of
                {ok, Text} -> {ok, Text};
                error -> {ok, JsonBin} %% Return raw if can't parse
            end
    end.

%% Extract array from JSON (simple regex-like approach)
extract_json_array(Bin) ->
    %% Find content between [ and ]
    case binary:match(Bin, <<"[">>) of
        {Start, 1} ->
            case binary:match(Bin, <<"]">>, [{scope, {Start, byte_size(Bin) - Start}}]) of
                {End, 1} ->
                    ArrayContent = binary:part(Bin, Start + 1, End - Start - 1),
                    parse_number_list(ArrayContent);
                nomatch ->
                    error
            end;
        nomatch ->
            error
    end.

%% Parse flat array of numbers
parse_flat_array(Bin) ->
    extract_json_array(Bin).

%% Parse comma-separated numbers
parse_number_list(Bin) ->
    Parts = binary:split(Bin, <<",">>, [global, trim_all]),
    Numbers = lists:filtermap(fun(Part) ->
        Trimmed = string:trim(binary_to_list(Part)),
        case string:to_float(Trimmed) of
            {F, []} -> {true, F};
            {error, _} ->
                case string:to_integer(Trimmed) of
                    {I, []} -> {true, float(I)};
                    _ -> false
                end;
            {F, _} -> {true, F}
        end
    end, Parts),
    case length(Numbers) > 0 of
        true -> {ok, Numbers};
        false -> error
    end.

%% Extract string value from JSON by key
extract_json_string(Bin, Key) ->
    %% Look for "key": "value" pattern
    Pattern = <<"\"", Key/binary, "\"">>,
    case binary:match(Bin, Pattern) of
        {KeyStart, KeyLen} ->
            %% Find the colon and opening quote
            Rest = binary:part(Bin, KeyStart + KeyLen, byte_size(Bin) - KeyStart - KeyLen),
            case binary:match(Rest, <<":">>) of
                {ColonPos, 1} ->
                    AfterColon = binary:part(Rest, ColonPos + 1, byte_size(Rest) - ColonPos - 1),
                    case binary:match(AfterColon, <<"\"">>) of
                        {QuoteStart, 1} ->
                            AfterQuote = binary:part(AfterColon, QuoteStart + 1, byte_size(AfterColon) - QuoteStart - 1),
                            case binary:match(AfterQuote, <<"\"">>) of
                                {EndQuote, 1} ->
                                    Value = binary:part(AfterQuote, 0, EndQuote),
                                    {ok, Value};
                                nomatch -> error
                            end;
                        nomatch -> error
                    end;
                nomatch -> error
            end;
        nomatch ->
            error
    end.
