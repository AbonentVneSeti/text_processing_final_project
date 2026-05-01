import numpy as np
import matplotlib.pyplot as plt
import json
import os
from .metrics import compute_metrics

def train_model(model, train_loader, val_loader, config, trainer_config, metrics_config):
    history = model.train(train_loader, val_loader, trainer_config)
    
    log_hist = [h for h in history if 'loss' in h and 'eval_loss' in h]
    if log_hist:
        steps = [h.get('step', i) for i, h in enumerate(log_hist)]
        train_loss = [h['loss'] for h in log_hist]
        eval_loss = [h['eval_loss'] for h in log_hist]
        plt.figure()
        plt.plot(steps, train_loss, label='Train Loss')
        plt.plot(steps, eval_loss, label='Eval Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.savefig(os.path.join(trainer_config['output_dir'], 'loss_curve.png'))

    with open(os.path.join(trainer_config['output_dir'], 'history.json'), 'w') as f:
        json.dump(history, f)
    print("Training completed. Model saved to", trainer_config['output_dir'])
    return model