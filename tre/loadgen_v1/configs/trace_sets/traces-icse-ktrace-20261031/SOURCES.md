# traces-icse-ktrace-20261031: sources and checksums (scale_icse_suite.py)

ICSE 10-30 suite (`traces-icse-20261030`, 7 traces) with only the arrival rate x a per-trace k (Alternating_hot_model_periodic_A 2.1, Decode_heavy_burst 1.2, Prefill_mixed_corner_decode_mix 1.15, Simultaneous_spike_ramp_twice_tps1o2 1.4, Sinusoidal_demand 1.3, Real_code_2024_slice_a_tok70 1.6, Real_conv_2023_slice_a_tok70 1.0). Trace files are never committed; the scripts live in the local workspace `docs/scripts/icse_suite_20261030/` (copy in `_src/`).

## Method

```
ICSE (v1 paper) suite with only the arrival rate scaled by k: <out>/<trace>/seed<s>/{traces_tre.effective.json,
manifest.json, preserve_history.json}, same layout as traces-icse-20261030 (build_icse_suite.py).

The 7 ICSE traces were made by v1's CustomTraceGenerator (aibrix-main CustomTraceGenerator/src); this wrapper runs that
code unchanged, with two patches applied in-process:
* prompts: DataGenerator.generate_prompt is called with randomness=0 (v1 passes 0.9, whose only effect is
  random.seed(None) before every prompt, which made v1 unreproducible); the stream is then seeded by
  config.random_seed = <ICSE config random_seed> + seed - 1. Prompt text style and length rule are v1's.
* counts: synthetic traces (custom mode): v1 sends round(rps_m(t)) requests in second t (rps from trace.json,
  3 s gaps linearly interpolated); that count reproduces every ICSE per-second count exactly. Scaled count =
  k x that ICSE count, rounded with carry per model (cumulative round), so the total is k x ICSE and each second is
  within 1 request of k x ICSE. (Multiplying trace.json rps by k and rounding per second is not used: rps values like
  0.4 / 3.0 / 11.75 round to a factor 1.17-1.62 instead of k.)
  Real_*: v1's real_trace mode replays per-second request lists (real_traces/<m>.json); ICSE's lists are the v9
  slice x v1_factor (v1 step 4, see tre_replayer/tracegen/realv1.py). Scaled list = ICSE's own requests + copies
  until round(v9 count x v1_factor x k) (the contend / realv1 count rule, rounded per second); each copy is a random
  v9 request of that second with input and output x U(0.85, 1.15) (v1 step 4 rule). k = 1 gives ICSE's lists.
Then: max_output_tokens as ICSE effective (make_effective_traces_0607.py: the 4 traces run on 06-07 send the config's
max_tokens for every request; the others keep the generator's value, null = config max_tokens).
PreServe history: --history icse (default, x1.3 / x1.5 sets): build_icse_suite.py's rows x k (synthetic: keyframe
rps at t0 x k; Real_*: v1 Azure counts x k, P / D = this trace's means). --history azure (10-31 ktrace set):
preserve_history_azure.build() = real Azure 2024 segments, levelled to the trace's planned mean (see that file);
it is given design inputs only, never the generated trace. Everything else (lengths, time shape, mix, caps,
duration) is ICSE's.
k: --k (one for all traces) or --k-map <json {trace: k}> (one per trace, user 10-31).

Usage (76, CPU only):  python3 scale_icse_suite.py (--k 1.5 | --k-map k.json) --out <root> --seeds 1 2 3
                         --v1-gen <CustomTraceGenerator> --v1-history-dir <tre_preserve/history> --v9 <traces_v9>
                         [--history icse|azure --azure-root <datasets/azure-llm-inference>] [--trace T ...]
                         [--processes 8]
```

Seeds: seed s uses v1 config random_seed = ICSE config random_seed + s - 1 (the ICSE files themselves were generated unseeded, so no seed reproduces them byte for byte; a k = 1 control run reproduces every ICSE (model, second, max_output_tokens, phase_type) multiset and the PreServe history exactly).

## Code

- `_src/audit_scaled.py` a48648287d83ca1a59e8db42151a8f43fd2669304d60984cfa54e155aa965b86
- `_src/build_icse_suite.py` c592e1f90b2e079a4e3bfe41c5872819b38cc83c97dacb2c4f38699bc9bad4fc
- `_src/check_history_causal.py` c02d0f1a5a6e99c458ee7d5a8767b0629ceabb44a0893fc3991828f64812f23f
- `_src/choose_k.py` a9c905b8962b01f277ec96a35f99c45eca5064e2c9863730784e85b3e7d2e563
- `_src/contend_lib.py` d1735ad34d3ed7995aae4b5d4b81d7f8a9c8e9e8e8946514a12da114e12c165c
- `_src/convert_icse_v2.py` 5184bcc8df21c3e77f1f0ac83c7c69cea044cd9589105e842af855efbbcd186b
- `_src/preserve_history_azure.py` 58335c9c8b2679ab0523edd5af429963d0ded21c0f049d372308a70789d8e37c
- `_src/scale_icse_suite.py` cb5dbbcf329facf728161e38455f9bc2389b02c0b5a22859ba0d28470fa50cbc
- `_src/verify_preserve_offline.py` 4d44329ff0ffb6acfabef583e406d75793b16e4eaa69a8fda67c39b490d9bf48
- v1 generator `/root/aibrix-main/CustomTraceGenerator` git aeb2cf42ba6c0c81680a918c3f7a98695498bafc (dirty src: none); `src/trace_generator.py` 459a8977a92650f393ed07b7e303ad498cfba89f2cc07d304bad8b9a4495795d, `src/data_generator.py` be2d06a61ba58ae44fc13ca9b52788640d9afbeb8d1e4ae55347f7d723af95d2, `src/real_trace_adapter.py` 6052126cd424a7779c45d81973f509513d6e7b6c9d294ee08bd0e9dd8b2d3bec, `src/config_manager.py` d9612f3864fd30d0335af1ea521e858cbaf6ed9b26f1584df9a9bb6fb78a8910

## Inputs

- Alternating_hot_model_periodic_A: `icse_final/Alternating_hot_model_periodic_A/config.yaml` 186e7d0f16fecfcc0d7c53ef842a246598fd07cb94f52136892062b248eca539, `icse_final/Alternating_hot_model_periodic_A/trace.json` 30833fd5cfc6cfcfdfd4ee4ad29bc6320a59087488456cacb28850b9546d2dff, loadgen config `traces_v14/Alternating_hot_model_periodic_A/config.yaml` c2de5bd4e02f41a8d07824464e28890a5e68e5fe4a2cb3ffbb9f4aa2b5410e67
- Decode_heavy_burst: `icse_final/Decode_heavy_burst/config.yaml` 519e6aced7e3d42ff40a4f4c1c1390190f708302317a21d0d3e4a833de3f55f4, `icse_final/Decode_heavy_burst/trace.json` a56deee1ec7d143b8aa6241cc96488b9373a0f8cf8d06632ad258af1b65acda1, loadgen config `traces_v14/Decode_heavy_burst/config.yaml` 436d890032f0933e8062b535b27d400e083c52dbe2583e0f4fdce5d3f69e2092
- Prefill_mixed_corner_decode_mix: `icse_final/Prefill_mixed_corner_decode_mix/config.yaml` b5b964694e08b0158f5a56c084b4f9342b44b31678c1057b8dd7d213c34ed562, `icse_final/Prefill_mixed_corner_decode_mix/trace.json` 9e7a5386990ccd0a842bcf13db7d1523b3ef0c2101b8e2a5be43fc2098d97f45, loadgen config `traces_v14/Prefill_mixed_corner_decode_mix/config.yaml` 5d8d8395ea8a3be9e2161694cea2cbe0385db543e68d4f90a45a0916ab47664e
- Simultaneous_spike_ramp_twice_tps1o2: `icse_final/Simultaneous_spike_ramp_twice_tps1o2/config.yaml` 782ef78f67e1bd816c8a67e0db7ad20c84fb422860aa194bb25d83591f074793, `icse_final/Simultaneous_spike_ramp_twice_tps1o2/trace.json` ed7a822f159c366e2d14430dacd1f8f39b0570f14b79b9e467a4a6cf4150f0ad, loadgen config `traces_v14/Simultaneous_spike_ramp_twice_tps1o2/config.yaml` 782ef78f67e1bd816c8a67e0db7ad20c84fb422860aa194bb25d83591f074793
- Sinusoidal_demand: `icse_final/Sinusoidal_demand/config.yaml` 471e040e2a5422e8f79c6e70d29e1626b8eb15708b98919aaaf87df7e65f49ba, `icse_final/Sinusoidal_demand/trace.json` b82890499e0a4548a5ba33d16cfabfaefc3e0f44600129f44cc966a95ad8ad39, loadgen config `traces_v14/Sinusoidal_demand/config.yaml` a13c6a033438a907e58a77c25c50c06d10eba86d25be783218a7ab9a358385e9
- Real_code_2024_slice_a_tok70: `icse_final/Real_code_2024_slice_a_tok70/config.yaml` 07a86578695d19bc28154c22945955c955061d3704d7fc0e0c44733df8451b29, `icse_final/Real_code_2024_slice_a_tok70/real_traces/dsllama-8b.json` 7a47be40acd69a46ef7ac5c82106116a5e75cfff8340cd7fb0aaa39c8e3e8750, `icse_final/Real_code_2024_slice_a_tok70/real_traces/dsqwen-7b.json` 8ce79625c4941c9cef6dd85f69805ca976e0ec20d7789665140685e18cce4a18, `icse_final/Real_code_2024_slice_a_tok70/real_traces/dsqwen-14b.json` 648886754563610fa67b6710aaeab341041b08224d32c0e728471c25ca1164a9, loadgen config `traces_v14/Real_code_2024_slice_a_tok70/config.yaml` 07a86578695d19bc28154c22945955c955061d3704d7fc0e0c44733df8451b29
  - v9 slice: `Real_code_2024_slice_a_tok70/dsllama-8b.json` 64ab73ce7f97d71a7d35e14188b7bdff603218d710e54545fc32f7ee1096ee3c, `Real_code_2024_slice_a_tok70/dsqwen-7b.json` afa52e72aedaa69978b8a7c56eb30f8ce8199048610273cec5f3d68d31595bf1, `Real_code_2024_slice_a_tok70/dsqwen-14b.json` 10c08a31350ccd644701121644e2aaf36e48e055bca97739ebe7239f00f3390c (`/data/nfs_shared_data/xxy/datasets/azure-llm-inference/v1-retrace/traces_v9`); v1 history `Real_code_2024_slice_a_tok70.json` 1e8116ed8e7ff7706959c13372fb71002c04c91add6df4890c30feb14bee2b23
- Real_conv_2023_slice_a_tok70: `icse_final/Real_conv_2023_slice_a_tok70/config.yaml` d6763533831d686685325906d8e7e911e5351aa419ef449e3e47e70ae34e0b29, `icse_final/Real_conv_2023_slice_a_tok70/real_traces/dsllama-8b.json` 73b28964c327384408f4b60334e5e188992fbbebcb085397a8c20a8a11135c4e, `icse_final/Real_conv_2023_slice_a_tok70/real_traces/dsqwen-7b.json` 335b21cefe318b5c19bd0ca5faada269288b149d28b0a6d90acb285ba27c430f, `icse_final/Real_conv_2023_slice_a_tok70/real_traces/dsqwen-14b.json` a30c63ecc85e6387e7dbc0404691e61f9ca1baba6d230248c869f34b92ed7115, loadgen config `traces_v14/Real_conv_2023_slice_a_tok70/config.yaml` d6763533831d686685325906d8e7e911e5351aa419ef449e3e47e70ae34e0b29
  - v9 slice: `Real_conv_2023_slice_a_tok70/dsllama-8b.json` 52a845224e53b041edf56bf4e03345f8d401bd3b006aa4b50968fbfe3cdb0f63, `Real_conv_2023_slice_a_tok70/dsqwen-7b.json` 7959b65c23699e43eba052033954a51f42f33929b1a7c17229f49a8da49a9363, `Real_conv_2023_slice_a_tok70/dsqwen-14b.json` 019dac2453130c74c739c420fa0a679c397874d38385deab49ddbf4f3b6130eb (`/data/nfs_shared_data/xxy/datasets/azure-llm-inference/v1-retrace/traces_v9`); v1 history `Real_conv_2023_slice_a_tok70.json` 0ad0768f5f3b1c1dc5584592757dc2a0a6e731c178ae0da84895af8940a92706

## Outputs

| trace/seed | requests | traces_tre.effective.json | preserve_history.json | manifest.json |
|---|---:|---|---|---|
| Alternating_hot_model_periodic_A/s1 | 25892 | 1d9e7b389b7ff42adacfa151cfb490eaa07fcc99d9b31fa9268668d6ffeaa9e5 | 7026a98c02cca0908cb17424aecb421622a36463ad3caf7673869ae41adeccea | bac62d4edd917723c40fa11de7c8546a4a93f1cbcadbf10f199ad4f4bd110302 |
| Alternating_hot_model_periodic_A/s2 | 25892 | 699a489aad5ed1e09ad9a45c5482fe431e2af9b154e0a735b3816b35c751c0a9 | bdc786b0254a85cfe703146787076c3872752fb060fc707151aff9a0dd78427a | cf6126d582d4f6ce8e0154df4e5f43188e03da5b4c1aa1ac7ff5d0b5c1c8fd3e |
| Alternating_hot_model_periodic_A/s3 | 25892 | 69c1c2e695fac84cef3f5e7e88316a1c558fa0ca263b2be33da520d8aaed2d54 | 9932d5ed5b13c60a4a9c78066482a80bac4ef95930bd3fba4bbf2ac368b03ce4 | fd85b293106da3090761e2e0da9292c6c6f5bf34bd14b27cf81596538ace1ee6 |
| Decode_heavy_burst/s1 | 26091 | f4ed951263fc94f14aef18355c6699671c7a261cfebf9b89e9f13b5f0ae05929 | e9ac0955144d5dc42fc0b14dc427d710d2bae1545788ea396c23ad690c232ef8 | 0276ff1fb984697046b6684439eed13ee198bc189f066424f485c2d99b5d4311 |
| Decode_heavy_burst/s2 | 26091 | 2470bbba3cea0a4b7f18710a84cd83cf9da7429bb64f92a5dec2eae1c0eabd2a | 7de6bebf90bce2b779d22ce74170c44149b16ee3d4e08107c7e443f7164f23d8 | 0a1a5e640ddf3287b7398e8dbbdf08a6650bec8e5f6d9ace4e85f2bd0d186cc4 |
| Decode_heavy_burst/s3 | 26091 | dd6122938292b75ea0a4080cd2155f7653b849307a4d1763a06798fb9ecfad50 | 430649810866ac1f93b2b1f705e706080ba9e88879dde84433aae4447f9a1190 | 29a2b34f10c39d0dc5c111b575c29125efe1bc93856fe2bc2ad4f360263a1fc0 |
| Prefill_mixed_corner_decode_mix/s1 | 11995 | 6bedee7300a16ee2eb2c7dcdab3f3537adcbc33a472119fd758e57098e84a610 | 92225f1a2439162e6987146ba797ce0a442193d15832f8c7767072d2d3f73bb0 | 1cf7e3c2fdee61af8e7ca912f35c91c92bd4073fab35da756176e06e487ae7e5 |
| Prefill_mixed_corner_decode_mix/s2 | 11995 | f36fdcd274c3f85fe42b01db51b5baef853479a1c47b65a0f97444be5fddeb5c | 290914f647388f888e7ef76db5fb1109d5a6d0c7de1f20d3fd00a1a587ad6c21 | ee8e29b7e86f94116af7f813cef6ad460cae425e233ae05ebb5abb800e9196f3 |
| Prefill_mixed_corner_decode_mix/s3 | 11995 | d65940ae3fbbb875bffbf73c2548654c37f801921e24bd1ba1fe599bf2bbe96f | aa27165d65f098f1b97e446b78571d2d7ec7f02879dea9cc79649b9e250bcdfe | ab6b1f113d343bc6ac3e36d45763601b45871770b70f14dd160fd5e75731f8cb |
| Real_code_2024_slice_a_tok70/s1 | 68764 | b115e279de7c6eda4ac0c65fcd2bdfdac677a75b3ea24c0298202d23a35c4e72 | 2aa33baa227270bc0091b09a9ba6faae15a9b2e39236d8913ed23f089518d34c | 149445ecd6325ddecb9bd186b8651f2ce4c63f2b1ea96f06a32d574805dda765 |
| Real_code_2024_slice_a_tok70/s2 | 68764 | 5b01a65c72fc8b611709d0fc8ffffe60dd548d873ac8c1e582a48be728069027 | 114127ad558f6d2b4ce70dfb6bdc8c91ad2f5fd6d35154d64617f2939a10ab4e | dc05192a83778a8c2e2fa0418017bff481b430ccfd37924304c102c24176b4a3 |
| Real_code_2024_slice_a_tok70/s3 | 68764 | 0cfe2be27a8148ab9b1daf4cfe72ea8d97de408154b19dccac4be32080100ba8 | 9dcd49baf120b7c3a8a722eb96fea93211546212a73896b2c1458275feab43c9 | a5866b120dc278ec5f02363ad0d2cc0253421eade33dba47775ca3b394149533 |
| Real_conv_2023_slice_a_tok70/s1 | 45244 | 8fd76a795a1f43fe9f1d96594873fa5009c176640fb4ba0b72ad77bc7a438543 | 0ce483fdcb6d8c9a7c032c3b2e457bcc8ab866d0749ee814c6d1984e47c6b77a | f8dba50f3907e23c1d88464502b73e4ad46a52a735e1fb6de5ca01b0ba0a027d |
| Real_conv_2023_slice_a_tok70/s2 | 45244 | aa2e00c0295d940482e5bab12c2cff603764481e80fbddaa2c473d40979c40fe | db528e866b53a96cf75d3b51a2d3f7f87e4b739ee7181f6e3ca4558668bb3fc9 | d202e108f4360143e2c195ec62c4fe4ecaac5a682f0eb7202595c067cf5a5ad8 |
| Real_conv_2023_slice_a_tok70/s3 | 45244 | 758442d5fab141c59cab4363b3e4ad984578332beff3b63506a56c79cba87d57 | 683a738b5a02781e5e343d70d064f2c0f1b1d380d9e131657ef8e12e3472aad5 | b11f91985e954537b18c4a248976d7526409d67a72029dd38ae5294e40174138 |
| Simultaneous_spike_ramp_twice_tps1o2/s1 | 7168 | 451bbec35b990c47e0ee1f9b53a3a5e0b5958aa03e4b0d3f0bc1d6e0f6cac912 | 64e0147e4bc96420be3222370f0a3c1a46ff45674aec91015a9f3512749e79a5 | 2590b633a3a3d6252fd2bcd083bcfb69856fed739b0a16c5f425942c2068d294 |
| Simultaneous_spike_ramp_twice_tps1o2/s2 | 7168 | 1a46790e97d2ac3f595b2cf9f2d77894b52e73cc1b36d8232ad3bbc369d976a9 | f673946629585ccbacb9cad8c751df7dffcd379105fcb40d71d3653f98d094b9 | 1ba630d15ade6fc4fe54972280addb5a1837b2c4ff3dce0b985246eac2b80832 |
| Simultaneous_spike_ramp_twice_tps1o2/s3 | 7168 | 779e301fb7476e0daef009c7f75fb199036ed10df00e8e2eb4d23a052c0ac6ad | 8cf8c08b631272772e6f983be10e32f5b11c1dd38afdb7cca080825bcd5ba2ff | 132ec030cac9464d5261e0edcb1c7d2d6643e2043308ac3af0c50c69272f5293 |
| Sinusoidal_demand/s1 | 28122 | 5e1e3233bd0abf319a6be545fd08edd64e39964342de3a5b16a0b44fbbbc10ab | 3530a2890e48afafa384e618749ab8f68f3d5647e8cc25222e4e295aa2b27e0b | dc7cfce30459612d320355d13afd7922cce52c85d4700542f486acae820cfaa4 |
| Sinusoidal_demand/s2 | 28122 | a2a0b036f5fed3a3290045bdfb75bc383643e1e02470aceb58baaab86ca9b7ad | 132558e9f9aed5200a284a8b9530897bdc01aec2a60c310be2c8837361126cdc | e2dc82f7d5bea24ca07b6dc636a9d9ea1425808e9ed49090f530752fb5d538cc |
| Sinusoidal_demand/s3 | 28122 | b68206a358665d291e0dff42e7a1d349328912f6ddd0ab359a42e0a0fbc6bfee | c8e238354904c9956c92a4075bddc639f7313704f656293975794d3b37cd002e | 1808fe6a050a8d5bba0c1df9fb36bfe1e240f53b0f1cfd3a7cd25052dab21438 |

## PreServe history (all traces): real Azure segments

```
PreServe history (preserve_history.json, tre-preserve-history-v1) from real Azure segments (user 10-31).

Rule (all 7 ICSE traces, every seed): per model, a 750 s segment (25 x 30 s windows, the PreServe Tier-1 input)
of real Azure 2024 arrivals with their lengths, taken from a time range that overlaps no Real_* slice, no
pre-slice history and no Azure interval any other TRE suite uses (EXCLUDE, 1 h guards). The history never
reads the evaluated trace: build() gets only design inputs (ICSE config / trace.json, v9 slice files, k, the
Azure CSVs); it is given no path inside the trace set (check_history_causal.py tests this).

* source family: dsllama-8b <- conv 2024 week, dsqwen-7b / dsqwen-14b <- code 2024 week (two different
  segments of the same file; they never overlap each other).
* segment start: sha256("icse-ktrace-history-v1|<trace>|seed<s>|<model>|<attempt>") -> offset in the week,
  rounded down to 30 s; the first attempt whose [start, start + 750 s) overlaps no EXCLUDE zone and no
  already chosen segment of the same file (same trace / seed) is used. Recorded in source.per_model.
* lengths: the v9 tok70 transform (generate_traces_v9.py rescale_trace_file): min-max of the segment's own
  ContextTokens / GeneratedTokens to the model's v9 token target (CODE / CONV_TOKEN_TARGETS of the family
  used), rounded, >= 1; then capped at the trace's planned max (input / output) for that model.
* level: P (prompt) and D (output) are scaled separately so that their mean per 30 s window equals the trace's
  PLANNED mean per 30 s window over the 720 s. Plan = the generator's per-second count schedule x planned
  lengths, never the effective file:
    synthetic: count(t) = carry-rounded k x round(rps_m(t)) (v1 timeline, scale_icse_suite.py rule);
               length(t) = mean of 4000 draws of the generator's own _get_request_overrides +
               _generate_prompt_length for the segment covering t (planning RNG seeded by trace / model),
               output = override max_tokens, else the config's max_tokens (06-07 traces: config max_tokens).
    Real_*:    count(t) = round(v9 count(t) x v1_factor x k) (scale_icse_suite.py rule);
               length(t) = mean v9 input / output tokens of second t (the added copies have jitter mean 1).
  The time shape and noise inside the 750 s stay the real segment's.
```

Excluded Azure ranges (UTC, both files):

- [2024-05-13T08:00:00+00:00, 2024-05-13T12:00:00+00:00): Real_conv_2023_slice_a..d (conv 2024 week [118800, 126000) s) + pre-slice history
- [2024-05-10T23:00:00+00:00, 2024-05-11T01:30:00+00:00): Real_code_2024_slice_a..d (code 2024 week [86400, 88200) s) + pre-slice history
- [2024-05-12T14:24:00+00:00, 2024-05-12T16:24:00+00:00): tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-12T15:24:00+00:00
- [2024-05-14T05:50:00+00:00, 2024-05-14T07:50:00+00:00): tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-14T06:50:00+00:00
- [2024-05-14T21:53:00+00:00, 2024-05-14T23:53:00+00:00): tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-14T22:53:00+00:00
- [2024-05-15T03:08:00+00:00, 2024-05-15T05:08:00+00:00): tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-15T04:08:00+00:00
- [2024-05-17T12:54:00+00:00, 2024-05-17T14:54:00+00:00): tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-17T13:54:00+00:00
- [2024-05-18T01:53:00+00:00, 2024-05-18T03:53:00+00:00): tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-18T02:53:00+00:00

| trace/seed | k | model | Azure segment (UTC) | raw req/s | scale P / D |
|---|---:|---|---|---:|---|
| Alternating_hot_model_periodic_A/s1 | 2.1 | dsllama-8b | conv 2024-05-13T23:34:30+00:00 .. 23:47:00 | 43.995 | 0.466734 / 1.096794 |
| Alternating_hot_model_periodic_A/s1 | 2.1 | dsqwen-7b | code 2024-05-12T14:00:00+00:00 .. 14:12:30 | 8.317 | 2.398038 / 8.607956 |
| Alternating_hot_model_periodic_A/s1 | 2.1 | dsqwen-14b | code 2024-05-14T10:45:30+00:00 .. 10:58:00 | 9.636 | 1.014278 / 3.515077 |
| Alternating_hot_model_periodic_A/s2 | 2.1 | dsllama-8b | conv 2024-05-14T15:47:00+00:00 .. 15:59:30 | 66.399 | 0.312599 / 0.740809 |
| Alternating_hot_model_periodic_A/s2 | 2.1 | dsqwen-7b | code 2024-05-15T17:43:30+00:00 .. 17:56:00 | 77.921 | 0.24184 / 0.953268 |
| Alternating_hot_model_periodic_A/s2 | 2.1 | dsqwen-14b | code 2024-05-10T11:39:30+00:00 .. 11:52:00 | 11.741 | 0.853251 / 2.893739 |
| Alternating_hot_model_periodic_A/s3 | 2.1 | dsllama-8b | conv 2024-05-13T02:17:00+00:00 .. 02:29:30 | 52.097 | 0.388104 / 0.905184 |
| Alternating_hot_model_periodic_A/s3 | 2.1 | dsqwen-7b | code 2024-05-10T15:37:00+00:00 .. 15:49:30 | 53.011 | 0.361157 / 1.397353 |
| Alternating_hot_model_periodic_A/s3 | 2.1 | dsqwen-14b | code 2024-05-11T03:04:00+00:00 .. 03:16:30 | 14.132 | 0.724609 / 2.310019 |
| Decode_heavy_burst/s1 | 1.2 | dsllama-8b | conv 2024-05-17T00:11:00+00:00 .. 00:23:30 | 44.141 | 0.306682 / 0.75 |
| Decode_heavy_burst/s1 | 1.2 | dsqwen-7b | code 2024-05-12T17:15:30+00:00 .. 17:28:00 | 11.715 | 1.057645 / 4.155608 |
| Decode_heavy_burst/s1 | 1.2 | dsqwen-14b | code 2024-05-15T00:19:00+00:00 .. 00:31:30 | 28.599 | 0.364558 / 1.247507 |
| Decode_heavy_burst/s2 | 1.2 | dsllama-8b | conv 2024-05-13T13:21:00+00:00 .. 13:33:30 | 69.841 | 0.195239 / 0.48285 |
| Decode_heavy_burst/s2 | 1.2 | dsqwen-7b | code 2024-05-15T11:21:00+00:00 .. 11:33:30 | 12.329 | 0.965594 / 3.793909 |
| Decode_heavy_burst/s2 | 1.2 | dsqwen-14b | code 2024-05-14T14:07:00+00:00 .. 14:19:30 | 62.937 | 0.164109 / 0.571436 |
| Decode_heavy_burst/s3 | 1.2 | dsllama-8b | conv 2024-05-17T03:07:30+00:00 .. 03:20:00 | 40.593 | 0.330455 / 0.803087 |
| Decode_heavy_burst/s3 | 1.2 | dsqwen-7b | code 2024-05-15T22:16:30+00:00 .. 22:29:00 | 40.621 | 0.300004 / 1.198161 |
| Decode_heavy_burst/s3 | 1.2 | dsqwen-14b | code 2024-05-10T11:50:00+00:00 .. 12:02:30 | 13.236 | 0.772293 / 2.60059 |
| Prefill_mixed_corner_decode_mix/s1 | 1.15 | dsllama-8b | conv 2024-05-15T06:05:00+00:00 .. 06:17:30 | 46.385 | 0.205745 / 0.756756 |
| Prefill_mixed_corner_decode_mix/s1 | 1.15 | dsqwen-7b | code 2024-05-12T07:49:00+00:00 .. 08:01:30 | 3.723 | 2.753502 / 16.078702 |
| Prefill_mixed_corner_decode_mix/s1 | 1.15 | dsqwen-14b | code 2024-05-11T02:00:30+00:00 .. 02:13:00 | 13.257 | 0.377609 / 2.120853 |
| Prefill_mixed_corner_decode_mix/s2 | 1.15 | dsllama-8b | conv 2024-05-13T12:36:30+00:00 .. 12:49:00 | 62.217 | 0.160383 / 0.581309 |
| Prefill_mixed_corner_decode_mix/s2 | 1.15 | dsqwen-7b | code 2024-05-14T05:06:30+00:00 .. 05:19:00 | 12.364 | 0.830867 / 5.094249 |
| Prefill_mixed_corner_decode_mix/s2 | 1.15 | dsqwen-14b | code 2024-05-11T23:18:00+00:00 .. 23:30:30 | 12.293 | 0.407198 / 2.239181 |
| Prefill_mixed_corner_decode_mix/s3 | 1.15 | dsllama-8b | conv 2024-05-13T05:26:30+00:00 .. 05:39:00 | 44.428 | 0.21605 / 0.783293 |
| Prefill_mixed_corner_decode_mix/s3 | 1.15 | dsqwen-7b | code 2024-05-15T07:27:00+00:00 .. 07:39:30 | 8.824 | 1.077617 / 6.857899 |
| Prefill_mixed_corner_decode_mix/s3 | 1.15 | dsqwen-14b | code 2024-05-10T05:12:30+00:00 .. 05:25:00 | 11.087 | 0.42756 / 2.462482 |
| Real_code_2024_slice_a_tok70/s1 | 1.6 | dsllama-8b | conv 2024-05-15T10:20:00+00:00 .. 10:32:30 | 51.859 | 0.90893 / 0.552723 |
| Real_code_2024_slice_a_tok70/s1 | 1.6 | dsqwen-7b | code 2024-05-16T12:37:30+00:00 .. 12:50:00 | 32.851 | 1.274517 / 1.224702 |
| Real_code_2024_slice_a_tok70/s1 | 1.6 | dsqwen-14b | code 2024-05-13T19:51:00+00:00 .. 20:03:30 | 81.256 | 0.160796 / 0.16191 |
| Real_code_2024_slice_a_tok70/s2 | 1.6 | dsllama-8b | conv 2024-05-17T23:07:00+00:00 .. 23:19:30 | 24.821 | 1.925881 / 1.170541 |
| Real_code_2024_slice_a_tok70/s2 | 1.6 | dsqwen-7b | code 2024-05-16T23:09:30+00:00 .. 23:22:00 | 28.604 | 1.473682 / 1.408791 |
| Real_code_2024_slice_a_tok70/s2 | 1.6 | dsqwen-14b | code 2024-05-13T16:58:30+00:00 .. 17:11:00 | 59.151 | 0.21948 / 0.223371 |
| Real_code_2024_slice_a_tok70/s3 | 1.6 | dsllama-8b | conv 2024-05-16T20:44:30+00:00 .. 20:57:00 | 44.856 | 1.080407 / 0.649147 |
| Real_code_2024_slice_a_tok70/s3 | 1.6 | dsqwen-7b | code 2024-05-10T07:42:00+00:00 .. 07:54:30 | 8.389 | 4.867053 / 4.753615 |
| Real_code_2024_slice_a_tok70/s3 | 1.6 | dsqwen-14b | code 2024-05-14T00:30:00+00:00 .. 00:42:30 | 28.883 | 0.462377 / 0.457144 |
| Real_conv_2023_slice_a_tok70/s1 | 1.0 | dsllama-8b | conv 2024-05-13T23:53:30+00:00 .. 00:06:00 | 45.66 | 0.522813 / 0.502039 |
| Real_conv_2023_slice_a_tok70/s1 | 1.0 | dsqwen-7b | code 2024-05-11T16:47:30+00:00 .. 17:00:00 | 15.869 | 1.349867 / 1.915967 |
| Real_conv_2023_slice_a_tok70/s1 | 1.0 | dsqwen-14b | code 2024-05-13T12:58:30+00:00 .. 13:11:00 | 28.716 | 0.457118 / 0.708429 |
| Real_conv_2023_slice_a_tok70/s2 | 1.0 | dsllama-8b | conv 2024-05-15T22:21:30+00:00 .. 22:34:00 | 34.9 | 0.698837 / 0.674589 |
| Real_conv_2023_slice_a_tok70/s2 | 1.0 | dsqwen-7b | code 2024-05-15T12:37:30+00:00 .. 12:50:00 | 30.725 | 0.668518 / 1.03165 |
| Real_conv_2023_slice_a_tok70/s2 | 1.0 | dsqwen-14b | code 2024-05-15T14:35:30+00:00 .. 14:48:00 | 59.053 | 0.221718 / 0.348012 |
| Real_conv_2023_slice_a_tok70/s3 | 1.0 | dsllama-8b | conv 2024-05-15T09:06:00+00:00 .. 09:18:30 | 54.48 | 0.44105 / 0.417621 |
| Real_conv_2023_slice_a_tok70/s3 | 1.0 | dsqwen-7b | code 2024-05-16T20:05:00+00:00 .. 20:17:30 | 78.965 | 0.255423 / 0.401793 |
| Real_conv_2023_slice_a_tok70/s3 | 1.0 | dsqwen-14b | code 2024-05-11T17:27:30+00:00 .. 17:40:00 | 15.584 | 0.894683 / 1.267165 |
| Simultaneous_spike_ramp_twice_tps1o2/s1 | 1.4 | dsllama-8b | conv 2024-05-16T11:57:00+00:00 .. 12:09:30 | 52.341 | 0.219842 / 0.383116 |
| Simultaneous_spike_ramp_twice_tps1o2/s1 | 1.4 | dsqwen-7b | code 2024-05-15T21:01:00+00:00 .. 21:13:30 | 69.36 | 0.132373 / 0.416733 |
| Simultaneous_spike_ramp_twice_tps1o2/s1 | 1.4 | dsqwen-14b | code 2024-05-16T05:21:30+00:00 .. 05:34:00 | 12.625 | 0.458196 / 1.213815 |
| Simultaneous_spike_ramp_twice_tps1o2/s2 | 1.4 | dsllama-8b | conv 2024-05-13T23:27:00+00:00 .. 23:39:30 | 43.345 | 0.262599 / 0.467263 |
| Simultaneous_spike_ramp_twice_tps1o2/s2 | 1.4 | dsqwen-7b | code 2024-05-10T09:48:30+00:00 .. 10:01:00 | 7.079 | 1.28713 / 3.807152 |
| Simultaneous_spike_ramp_twice_tps1o2/s2 | 1.4 | dsqwen-14b | code 2024-05-12T09:34:30+00:00 .. 09:47:00 | 2.856 | 1.956397 / 5.402443 |
| Simultaneous_spike_ramp_twice_tps1o2/s3 | 1.4 | dsllama-8b | conv 2024-05-17T04:02:30+00:00 .. 04:15:00 | 38.075 | 0.289848 / 0.507226 |
| Simultaneous_spike_ramp_twice_tps1o2/s3 | 1.4 | dsqwen-7b | code 2024-05-12T10:42:00+00:00 .. 10:54:30 | 3.631 | 2.533032 / 7.497928 |
| Simultaneous_spike_ramp_twice_tps1o2/s3 | 1.4 | dsqwen-14b | code 2024-05-14T04:52:30+00:00 .. 05:05:00 | 12.743 | 0.456188 / 1.17248 |
| Sinusoidal_demand/s1 | 1.3 | dsllama-8b | conv 2024-05-14T07:59:30+00:00 .. 08:12:00 | 58.624 | 0.35675 / 0.678008 |
| Sinusoidal_demand/s1 | 1.3 | dsqwen-7b | code 2024-05-14T10:13:00+00:00 .. 10:25:30 | 9.232 | 1.68296 / 5.07434 |
| Sinusoidal_demand/s1 | 1.3 | dsqwen-14b | code 2024-05-15T13:52:00+00:00 .. 14:04:30 | 54.921 | 0.237692 / 0.658892 |
| Sinusoidal_demand/s2 | 1.3 | dsllama-8b | conv 2024-05-18T11:23:00+00:00 .. 11:35:30 | 31.567 | 0.669484 / 1.263439 |
| Sinusoidal_demand/s2 | 1.3 | dsqwen-7b | code 2024-05-13T05:01:00+00:00 .. 05:13:30 | 11.405 | 1.413501 / 4.207993 |
| Sinusoidal_demand/s2 | 1.3 | dsqwen-14b | code 2024-05-12T01:31:00+00:00 .. 01:43:30 | 14.016 | 0.957484 / 2.470189 |
| Sinusoidal_demand/s3 | 1.3 | dsllama-8b | conv 2024-05-14T18:54:30+00:00 .. 19:07:00 | 65.141 | 0.326721 / 0.638713 |
| Sinusoidal_demand/s3 | 1.3 | dsqwen-7b | code 2024-05-11T07:09:00+00:00 .. 07:21:30 | 6.275 | 2.542405 / 7.582757 |
| Sinusoidal_demand/s3 | 1.3 | dsqwen-14b | code 2024-05-13T07:06:00+00:00 .. 07:18:30 | 8.433 | 1.534118 / 4.237836 |

Campaign: `trace_root` = this directory, `loadgen_config_root` = `${TRE_REPO}/loadgen_v1/configs/traces_v14` (unchanged: k changes no config field the client reads). Audit: `audit.md` / `audit.json` (audit_scaled.py).

## Choice of k, checks (10-31)

- k per trace (`_checks/k_map.json`) = the largest k on a 0.05 grid with corrected integer demand Gi' <= 7 in every 30 s
  window (t >= 30 s), `_src/choose_k.py` (`_checks/choose_k.json`). Corrected demand rho' = lambda / (kappa x mu), F = 1, kappa
  per ICSE family from the 10-30 calibration windows (tools/contend_calib_20261030 agg4.py, pooled violations > 5 %):
  synthetic 7b 1.2 / 8b 1.05 / 14b 1.3, Real 7b 0.85 / 8b 0.95 / 14b 1.2. With the r4 KAPPA_CAP (0.85 / 0.95 / 1.0) four
  ICSE traces already have Gi' = 8 at k = 1 (Decode_heavy, Prefill_mixed, Sinusoidal, Real_conv), so r4 kappa cannot pick k.
- Audit (`audit.md` / `audit.json`, `_src/audit_scaled.py --k-map`): request / token ratio = k within 2 %, Pearson r of the
  10 s curves > 0.99, lengths unchanged, corrected demand for both kappa sets, seeds 1-3: all pass.
- History causality (`_checks/check_history_causal.json`, `_src/check_history_causal.py`): no file inside this set is opened
  while the history is built (audit hook), and the 21 histories rebuilt with garbage traces_tre.effective.json / manifest.json
  are byte-identical. (The first static check flagged the prose "not the effective file" in the rule string; the check now
  flags file names only; rerun: clean.)
- PreServe offline (`_checks/verify_preserve_offline.json`, `_src/verify_preserve_offline.py`): the runner's prepare_preserve
  copies this preserve_history.json (no pre_t0.json exists, so it is the one the policy reads), PreServePolicy loads it
  (format / trace / seed / 25 windows), mLSTM window-0 decisions computed for every model.
