import numpy as np
import matplotlib.pyplot as plt
import json
import os

def train_model(model, train_loader, val_loader, config, trainer_config, metrics_config):
    try:
        history = model.train(train_loader, val_loader, trainer_config, metrics_config)
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем. Сохраняем модель...")
        output_dir = trainer_config.get("output_dir", "./saves")
        model.save(output_dir)
        print(f"Модель сохранена в {output_dir}")
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'state'):
            history = model.trainer.state.log_history
        else:
            history = []

    output_dir = trainer_config.get("output_dir", "./saves")

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
        plot_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(plot_path)
        print(f"График сохранён в {plot_path}")

    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print("История обучения сохранена в", history_path)
    return model