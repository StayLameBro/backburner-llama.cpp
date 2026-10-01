#pragma once
// infernet runtime toggles: a value comes from the JSON file at $INFERNET_TOGGLES (default /tmp/infernet-toggles.json) if the file
// has the key, else from the environment variable of the same name, else the default. The file is re-read when its mtime
// changes, so a benchmark can flip a feature between two requests to one running server (no relaunch, no launch-to-launch noise).
//   echo '{"LLAMA_SPEC_SAMPLE": 0}' > /tmp/infernet-toggles.json
double infernet_toggle(const char * name, double def);

#include <cstdint>
#include <cstdio>
// selector-calibration capture (LLAMA_SELECTOR_CAPTURE=<prefix>): the drafter writes <prefix>.draft.bin, the verify writes
// <prefix>.p.jsonl; records join on (round, j). Returns nullptr when capture is off.
FILE * infernet_capture_file(const char * suffix);
int64_t & infernet_capture_round();
