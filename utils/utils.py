import os
import torch
import json


def save_best_model(model, optimizer, epoch, iou, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iou_macro': iou,
    }, save_path)


def save_experiment_results(args, model_config, results, best_iou, save_dir, model_name):
    config_dict = vars(args).copy()
    config_dict['model_config'] = model_config

    final_output = {
        'config': config_dict,
        'best_iou_macro': best_iou,
        'results': results
    }

    file_path = os.path.join(save_dir, f'{model_name}.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"  >> [Finished] Results saved to {file_path}")

def print_best_results(best_overall_stats):
    print("\n" + "=" * 50)
    print(" [Best Performance]")
    print(f"  - Epoch: {best_overall_stats['epoch']}")
    print(f"  - Best mPA : {best_overall_stats['accuracy'] * 100:.2f}%")
    print(f"  - Best mIoU : {best_overall_stats['iou_macro'] * 100:.2f}%")
    print(f"  - Best Dice : {best_overall_stats['dice_macro'] * 100:.2f}%")
