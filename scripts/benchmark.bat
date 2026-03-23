python ifeval.py --model-root c:\data\models\Huggingface --models 1,2,3,4 --quant-list 1,2 --think 0 --temperature 0.7
python ceval.py --model-root c:\data\models\Huggingface --models 1,2,3,4 --quant-list 1,2 --think 0 --temperature 0.7
python mmlu_redux.py --model-root c:\data\models\Huggingface --models 1,2,3,4 --quant-list 1,2 --think 0 --temperature 0.7
