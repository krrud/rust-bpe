/* tslint:disable */
/* eslint-disable */
/**
* @returns {any}
*/
export function test_print(): any;
/**
*/
export class TokenizerWrapper {
  free(): void;
/**
* @param {any} vocabulary
* @param {any} merge_rules
*/
  constructor(vocabulary: any, merge_rules: any);
/**
* @param {number} index
* @returns {any}
*/
  getToken(index: number): any;
/**
* @param {string} token
* @returns {any}
*/
  getIndex(token: string): any;
/**
* @param {any} indices
* @returns {any}
*/
  getTokens(indices: any): any;
/**
* @param {any} tokens
* @returns {any}
*/
  getIndices(tokens: any): any;
/**
* @param {string} text
* @returns {any}
*/
  tokenize(text: string): any;
/**
* @param {any} indices
* @returns {string}
*/
  detokenize(indices: any): string;
/**
* @param {string} text
* @returns {string}
*/
  static cleanText(text: string): string;
/**
* @param {string} path
* @returns {any}
*/
  save(path: string): any;
/**
* @param {string} path
* @returns {any}
*/
  static load(path: string): any;
/**
* @param {any} sequences
* @param {number} max_len
* @returns {any}
*/
  padSequences(sequences: any, max_len: number): any;
/**
*/
  readonly getMergeRules: any;
/**
*/
  readonly getVocabulary: any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly test_print: () => number;
  readonly __wbg_tokenizerwrapper_free: (a: number) => void;
  readonly tokenizerwrapper_new: (a: number, b: number) => number;
  readonly tokenizerwrapper_getVocabulary: (a: number) => number;
  readonly tokenizerwrapper_getMergeRules: (a: number) => number;
  readonly tokenizerwrapper_getToken: (a: number, b: number) => number;
  readonly tokenizerwrapper_getIndex: (a: number, b: number, c: number) => number;
  readonly tokenizerwrapper_getTokens: (a: number, b: number) => number;
  readonly tokenizerwrapper_getIndices: (a: number, b: number) => number;
  readonly tokenizerwrapper_tokenize: (a: number, b: number, c: number) => number;
  readonly tokenizerwrapper_detokenize: (a: number, b: number, c: number) => void;
  readonly tokenizerwrapper_cleanText: (a: number, b: number, c: number) => void;
  readonly tokenizerwrapper_save: (a: number, b: number, c: number) => number;
  readonly tokenizerwrapper_load: (a: number, b: number) => number;
  readonly tokenizerwrapper_padSequences: (a: number, b: number, c: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
