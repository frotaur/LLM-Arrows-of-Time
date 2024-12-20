import torch,curses
from modules import get_tokenizer,MinGPT,MinGPT_Trainer
from torchenhanced import Trainer
import argparse, pathlib
import os

@torch.no_grad()
def display_completion(tokens, model:MinGPT, deTokenizer,gen_tokens, screen,backward):
    # Generate completion with the model
    full_tok = tokens
    y, x = 0, 0
    height, width = screen.getmaxyx()

    screen.clear()
    if(not backward):
        phrase = deTokenizer.detokenize(tokens) # Initial
        screen.addstr(y,x,phrase)
    else :
        phrase = deTokenizer.detokenize(tokens.flip(dims=[-1]))

    # screen.addstr(y,x,f'input token type : {full_tok.device}, model : {model.device} ')
    for _ in range(gen_tokens):
        new_tok = model.generate_next_token(full_tok,do_sample=True,top_k=50,temperature=0.8)
        # screen.addstr(y,x,str(new_tok))
        # time.sleep(3)
        full_tok = torch.cat([full_tok,new_tok],dim=-1)
        if(not backward):
            # Simply display tokens as they come
            y,x = screen.getyx()
            screen.addstr(deTokenizer.detokenize(new_tok))
            screen.refresh()
        else :
            # Clear and redisplay phrase so far, to get the 'backward' animation
            phrase=deTokenizer.detokenize(new_tok)+phrase
            screen.clear()
            screen.addstr(0,0,phrase)
            screen.refresh()
    y,x =screen.getyx()
    if(y+1==height):
        screen.scroll(1)
        screen.move(y,0)
    else:
        screen.move(y+1,0)
    screen.addstr("-"*(width-1))


def chat(screen,model,tokenizer,gen_tokens,backward,device):
    """ 
        Initiates a completion session with the model, displaying the tokens as they come.
        To be efficient with terminal writing (it was slowing down the generation),
        we use curses library. Can't use print() and input(), but use screen.addstr
        and screen.getstr (thanks ChatGPT)

        Args:
            screen : curses window object
            model : MinGPT model
            tokenizer : Tokenizer object
            gen_tokens : Number of tokens to generate
            backward : If the model is backwards
            device : Device to run the model on
    """
    curses.echo()
    curses.curs_set(1)
    screen.scrollok(True)  # Enable scrolling

    y,x = 0,0
    screen.addstr(y,x,f"Model loaded on {args.device}\n")
    while True:
        y,x = screen.getyx()
        if(backward):
            screen.addstr("\nText ending : ")
        else:
            screen.addstr("\nText beginning : ")
        screen.refresh()

        phrase = screen.getstr().decode("utf-8",errors='ignore') # Inputted phrase

        tokens = tokenizer.tokenize(phrase)
        if(backward):
            tokens = torch.flip(tokens,dims=[-1])
        tokens = torch.cat([torch.tensor([[0]]),tokens],dim=-1).to(device) # Add BOS token
        
        display_completion(tokens.to(device), model,tokenizer,gen_tokens=gen_tokens,screen=screen,backward = backward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate completions using a specified model.")
    parser.add_argument("model_path", help="Path of the save model, in .state format.")
    parser.add_argument("-d","--device",default="cpu", help="Specify the device to run on, e.g., 'cpu' or 'cuda'. Default is 'cpu'.")
    parser.add_argument("-g","--gen_tokens",default="128", help="Number of tokens to generate")
    parser.add_argument("-b", "--backward", action="store_true", help="Use it if it is a backwards model")
    parser.add_argument("-t", "--tokenizer_name", help="Name of the tokenizer to use. Use prefix like 'fr', 'en' or 'fi'")
    args = parser.parse_args()

    if(args.tokenizer_name is not None):
        tokenizer_path = pathlib.Path(__file__).parent.as_posix()
        tokenizer_path = os.path.join(tokenizer_path,'modules','tokenizers',f'{args.tokenizer_name}_tokenizer')
        tokenizer = get_tokenizer(m_path=tokenizer_path,m_name=args.tokenizer_name)
    else :
        raise ValueError("Tokenizer name must be specified")

    backward=args.backward

    # Load the model using the specified path
    # name, config, weights = MinGPT_Trainer.model_config_from_state(args.model_path,device=args.device)
    model = Trainer.get_model_from_state(MinGPT,args.model_path)
    model.to(args.device)
    # assert name=='MinGPT', 'For now only works with MinGPT models'

    # model = MinGPT(**config).to(args.device)
    # model.load_state_dict(weights,strict=True)
    model.eval()




    curses.wrapper(lambda s: chat(s,model,tokenizer,gen_tokens=int(args.gen_tokens),backward=backward, device=args.device))
