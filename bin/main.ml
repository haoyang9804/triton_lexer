module Lexer = Triton_lexer.Lexer
module Parser = Triton_lexer.Parser
module Converter = Triton_lexer.Converter
open Yojson.Basic.Util

let print_tokens tokens =
  print_endline "Token names:";
  print_endline "----------------";
  List.iter (fun t -> Printf.printf "%s\n" (Converter.string_of_token t)) tokens;
  print_endline "\nToken values:";
  print_endline "----------------";
  List.iter (fun t -> Printf.printf "%d\n" (Converter.token_to_int t)) tokens

let tokenize_string s =
  print_endline "Tokenizing string:";
  print_endline "----------------";
  (* let lexbuf = Lexing.from_string s in *)
  let tokens = Lexer.from_string s in
  print_tokens tokens;
  tokens
  (* try
    let token_seq = Parser.tokensLexer.token lexbuf in
    print_tokens token_seq;
    token_seq
  with
    | Parser.Error ->
        let pos = Lexing.lexeme_start_p lexbuf in
        Printf.printf "Parser error at line %d, column %d\n"
          pos.Lexing.pos_lnum
          (pos.Lexing.pos_cnum - pos.Lexing.pos_bol + 1);
        [] *)
let tokenize_file filename =
  print_endline ("Tokenizing file: " ^ filename);
  print_endline "----------------";
  let ic = open_in filename in
  let content = really_input_string ic (in_channel_length ic) in
  close_in ic;
  let tokens = Lexer.from_string content in
  print_tokens tokens;
  print_endline "----------------";
  tokens

let process_json_file filename =
  print_endline ("Processing JSON file: " ^ filename);
  print_endline "----------------";
  let json = Yojson.Basic.from_file filename in
  let kernels = json |> member "kernels" |> to_list in
  let processed_kernels = List.map (fun kernel ->
    let code = kernel |> member "code" |> to_string in
    let tokens = tokenize_string code in
    let token_strings = List.map Converter.string_of_token tokens in
    let token_values = List.map Converter.token_to_int tokens in
    let token_positions = List.mapi (fun i _ -> i) tokens in
    `Assoc [
      ("code", `String code);
      ("tokens", `List (List.map (fun s -> `String s) token_strings));
      ("token_values", `List (List.map (fun i -> `Int i) token_values));
      ("token_positions", `List (List.map (fun i -> `Int i) token_positions))
    ]
  ) kernels in
  let output_json = `Assoc [("kernels", `List processed_kernels)] in
  let output_filename = Filename.chop_extension filename ^ "_tokens.json" in
  Yojson.Basic.to_file output_filename output_json;
  print_endline ("Saved tokens to: " ^ output_filename);
  print_endline "----------------"

let usage () =
  print_endline "Usage:";
  print_endline "  ./main.exe <filename>  - Tokenize the given file";
  print_endline "  ./main.exe -s <string> - Tokenize the given string";
  print_endline "  ./main.exe -j <json>   - Process kernels from JSON file";
  exit 1

let () =
  match Array.length Sys.argv with
  | 2 -> ignore (tokenize_file Sys.argv.(1))
  | 3 when Sys.argv.(1) = "-s" -> ignore (tokenize_string Sys.argv.(2))
  | 3 when Sys.argv.(1) = "-j" -> process_json_file Sys.argv.(2)
  | _ -> usage ()
