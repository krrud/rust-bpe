/* tslint:disable */
/* eslint-disable */
/**
*/
export class TokenizerJs {
  free(): void;
/**
* @param {any} vocabulary
* @param {any} merge_rules
* @param {any} config
*/
  constructor(vocabulary: any, merge_rules: any, config: any);
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
*/
  readonly getMergeRules: any;
/**
*/
  readonly getVocabulary: any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_tokenizerjs_free: (a: number) => void;
  readonly tokenizerjs_new: (a: number, b: number, c: number) => number;
  readonly tokenizerjs_getVocabulary: (a: number) => number;
  readonly tokenizerjs_getMergeRules: (a: number) => number;
  readonly tokenizerjs_getToken: (a: number, b: number) => number;
  readonly tokenizerjs_getIndex: (a: number, b: number, c: number) => number;
  readonly tokenizerjs_getTokens: (a: number, b: number) => number;
  readonly tokenizerjs_getIndices: (a: number, b: number) => number;
  readonly tokenizerjs_tokenize: (a: number, b: number, c: number) => number;
  readonly tokenizerjs_detokenize: (a: number, b: number, c: number) => void;
  readonly tokenizerjs_cleanText: (a: number, b: number, c: number) => void;
  readonly tokenizerjs_save: (a: number, b: number, c: number) => number;
  readonly tokenizerjs_load: (a: number, b: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
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
