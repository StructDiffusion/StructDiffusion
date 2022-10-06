from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

# [top]

# Put the [objects] in a [size][shape] on the [x][y] of the table facing [rotation].
# Build a [size][shape] of the [objects] on the [x][y] of the table facing [rotation].
# Put the [objects] on the [x][y] of the table and make a [shape] facing [rotation].
# Rearrange the [objects] into a [shape], and put the structure on the [x][y] of the table facing [rotation].
# Could you ...
# Please ...
# Pick up the objects, put them into a [size][shape], place the [shape] on the [x][y] of table, make sure the [shape] is facing [rotation].

if __name__ == "__main__":
    ''' 
    uncomment to get reproducable paraphrase generations
    def random_state(seed):
      torch.manual_seed(seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    random_state(1234)
    '''

    #Init models (make sure you init ONLY once if you integrate this to your code)
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    phrases = ["Rearrange the mugs in a circle on the top left of the table."]

    for phrase in phrases:
      print("-"*100)
      print("Input_phrase: ", phrase)
      print("-"*100)
      para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False, max_return_phrases=100, do_diverse=True)
      for para_phrase in para_phrases:
       print(para_phrase)