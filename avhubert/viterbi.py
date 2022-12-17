from typing import Any, Dict, List
import torch
from fairseq.models.fairseq_model import FairseqModel
from examples.speech_recognition.new.decoders.base_decoder import BaseDecoder


class ViterbiDecoder(BaseDecoder):
    def get_emissions(
        self,
        models: List[FairseqModel],
        encoder_input: Dict[str, Any],
    ) -> torch.FloatTensor:
        model = models[0]
        encoder_out = model(**encoder_input)  # (B, T, V)
        emissions = torch.nn.functional.log_softmax(encoder_out, dim=-1)
        return emissions.float().cpu().contiguous() # (B, T, V)
    
    def decode(
        self,
        emissions: torch.FloatTensor, # (B, T, V)
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        def get_pred(e):
            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != self.blank]

        return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]