//// COBS (Consistent Overhead Byte Stuffing) implementation for VIVA-Link
////
//// COBS removes all 0x00 bytes from data, allowing 0x00 to be used as
//// an unambiguous packet delimiter. Overhead: max 1 byte per 254 bytes.
////
//// Reference: https://en.wikipedia.org/wiki/Consistent_Overhead_Byte_Stuffing

import gleam/list

/// Packet delimiter (always 0x00 after COBS encoding)
pub const delimiter = 0x00

// ============================================================================
// Erlang FFI for binary <-> list conversion
// ============================================================================

@external(erlang, "erlang", "binary_to_list")
fn binary_to_list(data: BitArray) -> List(Int)

@external(erlang, "erlang", "list_to_binary")
fn list_to_binary(data: List(Int)) -> BitArray

// ============================================================================
// Public API
// ============================================================================

/// Encode data using COBS algorithm
/// Returns encoded data WITH trailing 0x00 delimiter
pub fn encode(data: BitArray) -> BitArray {
  let bytes = binary_to_list(data)
  let encoded = do_encode(bytes)
  list_to_binary(list.append(encoded, [delimiter]))
}

/// Decode COBS-encoded data (without trailing delimiter)
/// Returns Ok(decoded) or Error if invalid encoding
pub fn decode(data: BitArray) -> Result(BitArray, String) {
  let bytes = binary_to_list(data)
  case do_decode(bytes) {
    Ok(decoded) -> Ok(list_to_binary(decoded))
    Error(e) -> Error(e)
  }
}

/// Extract complete frames from a buffer
/// Returns #(remaining_buffer, list_of_decoded_packets)
pub fn extract_frames(buffer: BitArray) -> #(BitArray, List(BitArray)) {
  let bytes = binary_to_list(buffer)
  let #(remaining, frames) = split_on_delimiter(bytes, [], [])
  #(list_to_binary(remaining), frames)
}

// ============================================================================
// COBS Encoder
// ============================================================================

fn do_encode(input: List(Int)) -> List(Int) {
  // Split input into blocks separated by zeros
  let blocks = split_by_zero(input, [], [])
  // Encode each block
  encode_blocks(blocks, [])
}

/// Split input list by zero bytes
/// Returns list of blocks (each block is data between zeros)
fn split_by_zero(
  input: List(Int),
  current_block: List(Int),
  blocks: List(List(Int)),
) -> List(List(Int)) {
  case input {
    [] -> {
      // End of input - add final block (even if empty means trailing zero)
      list.reverse([list.reverse(current_block), ..blocks])
    }
    [0, ..rest] -> {
      // Found zero - save current block and start new one
      split_by_zero(rest, [], [list.reverse(current_block), ..blocks])
    }
    [byte, ..rest] -> {
      // Normal byte - add to current block
      split_by_zero(rest, [byte, ..current_block], blocks)
    }
  }
}

/// Encode blocks into COBS format
fn encode_blocks(blocks: List(List(Int)), output: List(Int)) -> List(Int) {
  case blocks {
    [] -> list.reverse(output)
    [block] -> {
      // Last block - no implicit zero after
      let encoded = encode_single_block(block, True)
      list.reverse(list.append(list.reverse(encoded), output))
    }
    [block, ..rest] -> {
      // Not last block - implicit zero follows
      let encoded = encode_single_block(block, False)
      encode_blocks(rest, list.append(list.reverse(encoded), output))
    }
  }
}

/// Encode a single block (handles blocks > 254 bytes)
fn encode_single_block(block: List(Int), is_last: Bool) -> List(Int) {
  // is_last is used in recursive call for proper termination
  let _ = is_last
  case list.length(block) {
    0 -> {
      // Empty block = just a zero in original
      [1]
      // Code byte pointing to implicit zero
    }
    len if len < 254 -> {
      // Block fits in one chunk
      [len + 1, ..block]
    }
    _ -> {
      // Block too long - split at 254
      let #(chunk, rest) = list.split(block, 254)
      // 0xFF means 254 bytes follow, no implicit zero
      let chunk_encoded = [0xFF, ..chunk]
      list.append(chunk_encoded, encode_single_block(rest, is_last))
    }
  }
}

// ============================================================================
// COBS Decoder
// ============================================================================

fn do_decode(input: List(Int)) -> Result(List(Int), String) {
  decode_loop(input, [], True)
}

fn decode_loop(
  input: List(Int),
  output: List(Int),
  _first: Bool,
) -> Result(List(Int), String) {
  case input {
    [] -> Ok(list.reverse(output))

    [0, ..] -> Error("Invalid COBS: unexpected zero in encoded data")

    [code, ..rest] -> {
      let block_len = code - 1
      case take_exactly(rest, block_len) {
        Error(_) -> Error("Invalid COBS: truncated block")
        Ok(#(block, remaining)) -> {
          // Add the block data
          let new_output = list.append(list.reverse(block), output)

          // If code < 0xFF and there's more data, insert a zero
          let final_output = case code < 0xFF, remaining {
            True, [_, ..] -> [0, ..new_output]
            _, _ -> new_output
          }

          decode_loop(remaining, final_output, False)
        }
      }
    }
  }
}

fn take_exactly(input: List(a), n: Int) -> Result(#(List(a), List(a)), Nil) {
  take_exactly_acc(input, n, [])
}

fn take_exactly_acc(
  input: List(a),
  n: Int,
  acc: List(a),
) -> Result(#(List(a), List(a)), Nil) {
  case n, input {
    0, rest -> Ok(#(list.reverse(acc), rest))
    _, [] -> Error(Nil)
    _, [x, ..rest] -> take_exactly_acc(rest, n - 1, [x, ..acc])
  }
}

// ============================================================================
// Frame Extraction
// ============================================================================

fn split_on_delimiter(
  input: List(Int),
  current: List(Int),
  frames: List(BitArray),
) -> #(List(Int), List(BitArray)) {
  case input {
    [] -> #(list.reverse(current), list.reverse(frames))

    [0x00, ..rest] -> {
      // Found delimiter, decode current frame
      let frame_bytes = list.reverse(current)
      case frame_bytes {
        [] -> split_on_delimiter(rest, [], frames)
        // Empty frame, skip
        _ -> {
          case decode(list_to_binary(frame_bytes)) {
            Ok(decoded) -> split_on_delimiter(rest, [], [decoded, ..frames])
            Error(_) -> split_on_delimiter(rest, [], frames)
            // Skip invalid
          }
        }
      }
    }

    [byte, ..rest] -> {
      split_on_delimiter(rest, [byte, ..current], frames)
    }
  }
}
