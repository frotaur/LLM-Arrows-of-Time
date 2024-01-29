# BackPerplexity
Investigate perplexity of LLM's when trained backward vs forward

# Testing model generation
I made a little script (generate.py) to test the model generation. Example of use :


`python generate.py -b -g 256 -d "cuda:0" "GPT-124M_backwards"`

This will open an interface where you can type the starting (or ending) tokens, that will be completed by the model.

Generally :
`python generate.py [-b] [-g NUM_TOKENS] [-d DEVICE] MODEL_NAME`
The parameters are as follows :
- `-b` : include it if you want backward token generation (usually for a model trained backward)
- `-g <int>` : number of tokens to generate. Default 128.
- `-d <device>` : device on which to do the generation. Default 'cpu'.
- `MODEL_NAME` : name of the model as saved by the trainer. Saved states can be found in `runs/BackPerplexity/state/`. For example, for `GPT-5M.state`, the `MODEL_NAME` would be `'GPT-5M'`.

Don't forget to upgrade `torchenhanced` to the latest version ! (see requirements.txt)