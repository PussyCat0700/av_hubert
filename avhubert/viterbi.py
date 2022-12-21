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
        with torch.no_grad():
            output = model.encoder(**encoder_input)  # (T, B, C), (B, T), (B, T)
            encoder_out = output['encoder_out'].transpose(0, 1).transpose(1, 2)  # (L,B,dmodel)->(B,dmodel,L)
            encoder_out = model.ctc_output_conv(encoder_out)  # (B,V,L)
            encoder_out = encoder_out.transpose(1, 2)  # (B,L,V)
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