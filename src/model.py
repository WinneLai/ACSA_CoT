import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path, load_in_8bit=True, device_map = 'auto')
        # Define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        # prepare int-8 model for training
        self.engine = prepare_model_for_int8_training(self.engine)

        # add LoRA adaptor
        self.engine = get_peft_model(self.engine, lora_config)
        self.engine.print_trainable_parameters()
        # load_checkpoint_and_dispatch(
        #         self.engine,
        #         config.model_path,
        #         device_map=self.engine.hf_device_map,
        #         offload_folder=None,
        #         dtype='float16',
        #         offload_state_dict=True
        #         )
        # max_memory = get_balanced_memory(
        #                 self.engine,
        #                 max_memory=None,
        #                 no_split_module_classes=["T5Block"],
        #                 dtype='float16',
        #                 low_zero=False,
        #             )
        # max_memory={0: "10GiB", 1: "24GiB", "cpu": "30GiB"}
        # device_map = infer_auto_device_map(
        #                 self.engine,
        #                 max_memory=max_memory,
        #                 no_split_module_classes=["T5Block"],
        #                 dtype='float16'
        #             )
        # device_map['lm_head'] = device_map["decoder.embed_tokens"]
        # self.engine = dispatch_model(self.engine, device_map=device_map)
        # for i in self.engine.named_parameters(): print(f"{i[0]} -> {i[1].device}")
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_path)

    def forward(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
        input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                      max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        return output
