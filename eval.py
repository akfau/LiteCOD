# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from tqdm import tqdm
# pip install pysodmetrics
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

# Ablation methods to evaluate
ablation_methods = ['baseline_attention', 'baseline_ecg', 'baseline_ecg_attention', 'baseline_ecg_local_global', 'baseline_global', 'baseline_local']

# Datasets to evaluate on
datasets = ['COD10K']  # You can add 'CAMO', 'NC4K' later

for method in ablation_methods:
    print(f"\n{'='*60}")
    print(f"Evaluating method: {method}")
    print(f"{'='*60}")
    
    for _data_name in datasets:
        print(f"\nEvaluating on dataset: {_data_name}")
        
        mask_root = 'D:/BCNet/data/Testset/{}/GT'.format(_data_name)
        pred_root = './test_results/{}/{}/'.format(method, _data_name)
        
        # Check if prediction directory exists
        if not os.path.exists(pred_root):
            print(f"Prediction directory not found: {pred_root}")
            print(f"Skipping {method} on {_data_name}")
            continue
        
        # Check if mask directory exists
        if not os.path.exists(mask_root):
            print(f"Mask directory not found: {mask_root}")
            continue
        
        mask_name_list = sorted(os.listdir(mask_root))
        
        # Check if there are predictions available
        pred_files = os.listdir(pred_root)
        if len(pred_files) == 0:
            print(f"No prediction files found in {pred_root}")
            continue
        
        print(f"Found {len(mask_name_list)} ground truth masks")
        print(f"Found {len(pred_files)} prediction files")
        
        # Initialize metrics
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        
        processed_count = 0
        missing_count = 0
        
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list), desc=f"{method}-{_data_name}"):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            
            # Check if both files exist
            if not os.path.exists(mask_path):
                print(f"Mask not found: {mask_path}")
                missing_count += 1
                continue
                
            if not os.path.exists(pred_path):
                # Try with different extensions
                base_name = os.path.splitext(mask_name)[0]
                pred_path_jpg = os.path.join(pred_root, base_name + '.jpg')
                pred_path_png = os.path.join(pred_root, base_name + '.png')
                
                if os.path.exists(pred_path_jpg):
                    pred_path = pred_path_jpg
                elif os.path.exists(pred_path_png):
                    pred_path = pred_path_png
                else:
                    missing_count += 1
                    continue
            
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is None:
                    print(f"Failed to load mask: {mask_path}")
                    missing_count += 1
                    continue
                    
                if pred is None:
                    print(f"Failed to load prediction: {pred_path}")
                    missing_count += 1
                    continue
                
                # Resize prediction to match mask if needed
                if mask.shape != pred.shape:
                    pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))
                
                FM.step(pred=pred, gt=mask)
                WFM.step(pred=pred, gt=mask)
                SM.step(pred=pred, gt=mask)
                EM.step(pred=pred, gt=mask)
                M.step(pred=pred, gt=mask)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {mask_name}: {e}")
                missing_count += 1
                continue
        
        print(f"Processed: {processed_count}/{len(mask_name_list)} images")
        print(f"Missing/Failed: {missing_count} images")
        
        if processed_count == 0:
            print(f"No images processed for {method} on {_data_name}")
            continue
        
        # Get results
        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]
        
        results = {
            "Method": method,
            "Dataset": _data_name,
            "Processed": f"{processed_count}/{len(mask_name_list)}",
            "Smeasure": round(sm, 4),
            "wFmeasure": round(wfm, 4),
            "MAE": round(mae, 4),
            "adpEm": round(em["adp"], 4),
            "meanEm": round(em["curve"].mean(), 4),
            "maxEm": round(em["curve"].max(), 4),
            "adpFm": round(fm["adp"], 4),
            "meanFm": round(fm["curve"].mean(), 4),
            "maxFm": round(fm["curve"].max(), 4),
        }
        
        print(f"\nResults for {method} on {_data_name}:")
        print("-" * 50)
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # Save results to file
        file = open("ablation_evaluation_results.txt", "a")
        file.write(f"{method} {_data_name} {str(results)}\n")
        file.close()
        
        # Also save in a more readable format
        readable_file = open("ablation_results_readable.txt", "a")
        readable_file.write(f"\n{method} - {_data_name}:\n")
        readable_file.write(f"  Sm: {results['Smeasure']}\n")
        readable_file.write(f"  wFm: {results['wFmeasure']}\n")
        readable_file.write(f"  MAE: {results['MAE']}\n")
        readable_file.write(f"  maxEm: {results['maxEm']}\n")
        readable_file.write(f"  adpEm: {results['adpEm']}\n")
        readable_file.close()

print(f"\n{'='*60}")
print("ABLATION EVALUATION COMPLETED!")
print("Results saved to:")
print("  - ablation_evaluation_results.txt (raw format)")
print("  - ablation_results_readable.txt (readable format)")
# print(f"{'='*60}))