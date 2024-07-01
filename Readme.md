# Byte-Pair Encoding Tokenizer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


## Overview <a name="overview"></a>
This project is a lightweight byte-pair encoding tokenizer capable of effectively modeling any language. My implementation was trained on a selection of texts from the Gutenberg Project, and is focused on the English language. The vocabulary is around 15,000 tokens and has proven to be able to adapt well to unseen text inputs.

Future improvements include a GPU training solution leveraging WGPU to allow for a larger training corpus, and more responsive training. I would also like to train a capitalized, and multi-lingual tokenizer after perfomance improvements are in place.


## Installation <a name="installation"></a>
To run this project, run the following commands in your terminal:

```sh
git clone https://github.com/krrud/rust-bpe.git
cd rust-bpe
cargo build
```


## Usage <a name="usage"></a>
**To train the tokenizer with your corpus:**
```rust
let corpus = "path/to/corpus/dir";
let output = "path/to/output-file.json";
let iterations = 5000;
let tokenizer = Tokenizer::train_cpu(corpus, iterations, Some(output));
```

**To tokenize text and retrieve tokens or indices:**
```rust
let tokenizer = Tokenizer::load("path/to/your/trained-tokenizer.json").unwrap();
let tokens = tokenizer.tokenize("your text to tokenize");
let token_vals = tokenizer.get_tokens(&tokens);
let token_indices = tokenizer.get_indices(&token_vals);
```

**To convert tokens back to their original string run the following command:**
```rust
let detokenized = tokenizer.detokenize(&tokens);
```


## License <a name="license"></a>
This project is licensed under the MIT License - see the LICENSE.md file for details.