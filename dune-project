(lang dune 3.15)
(using menhir 3.0)
(name triton_lexer)

(generate_opam_files true)

(source
 (github username/reponame))

(authors "Author Name")

(maintainers "Maintainer Name")

(license LICENSE)

(documentation https://url/to/documentation)

(package
 (name triton_lexer)
 (synopsis "A short synopsis")
 (description "A longer description")
 (depends
  ocaml
  dune
  yojson)
 (tags
  (topics "to describe" your project)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
