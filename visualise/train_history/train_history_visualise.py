import json
import matplotlib.pyplot as plt

with open('models/rut5_paraphraser/saves/history.json') as f:
    history = json.load(f)

train_entries = []
eval_entries = []
for entry in history:
    if 'loss' in entry and 'eval_loss' not in entry:
        train_entries.append(entry)
    elif 'eval_loss' in entry:
        eval_entries.append(entry)

epochs_train = [e['epoch'] for e in train_entries]
epochs_eval  = [e['epoch'] for e in eval_entries]

train_loss = [e['loss'] for e in train_entries]
eval_loss  = [e['eval_loss'] for e in eval_entries]
bleu       = [e['eval_bleu'] for e in eval_entries]
bertscore  = [e['eval_bertscore'] for e in eval_entries]
cosine     = [e['eval_cosine_similarity'] for e in eval_entries]

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

axs[0,0].plot(epochs_train, train_loss, 'o-', label='Train Loss')
axs[0,0].plot(epochs_eval, eval_loss, 's-', label='Eval Loss')
axs[0,0].set_xlabel('Epoch')
axs[0,0].set_ylabel('Loss')
axs[0,0].set_title('ruT5 – Loss')
axs[0,0].legend()
axs[0,0].grid(True)

axs[0,1].plot(epochs_eval, bleu, 'o-', color='green')
axs[0,1].set_xlabel('Epoch')
axs[0,1].set_ylabel('BLEU')
axs[0,1].set_title('ruT5 – BLEU')
axs[0,1].grid(True)

axs[1,0].plot(epochs_eval, bertscore, 'o-', color='orange')
axs[1,0].set_xlabel('Epoch')
axs[1,0].set_ylabel('BERTScore')
axs[1,0].set_title('ruT5 – BERTScore')
axs[1,0].grid(True)

axs[1,1].plot(epochs_eval, cosine, 'o-', color='red')
axs[1,1].set_xlabel('Epoch')
axs[1,1].set_ylabel('Cosine Similarity')
axs[1,1].set_title('ruT5 – Cosine Similarity')
axs[1,1].grid(True)

plt.tight_layout()
plt.savefig('rut5_training_metrics.png', dpi=150)
plt.show()




with open('models/simple_seq2seq/saves/history.json') as f:
    history_seq = json.load(f)

epochs = [h['epoch'] for h in history_seq]
train_loss = [h['loss'] for h in history_seq]
eval_loss  = [h['eval_loss'] for h in history_seq]
bleu       = [h['val_bleu'] for h in history_seq]
bertscore  = [h['val_bertscore'] for h in history_seq]
cosine     = [h['val_cosine_similarity'] for h in history_seq]

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

axs[0,0].plot(epochs, train_loss, 'o-', label='Train Loss')
axs[0,0].plot(epochs, eval_loss, 's-', label='Eval Loss')
axs[0,0].set_xlabel('Epoch')
axs[0,0].set_ylabel('Loss')
axs[0,0].set_title('Simple Seq2Seq – Loss')
axs[0,0].legend()
axs[0,0].grid(True)

axs[0,1].plot(epochs, bleu, 'o-', color='green')
axs[0,1].set_xlabel('Epoch')
axs[0,1].set_ylabel('BLEU')
axs[0,1].set_title('Simple Seq2Seq – BLEU')
axs[0,1].grid(True)

axs[1,0].plot(epochs, bertscore, 'o-', color='orange')
axs[1,0].set_xlabel('Epoch')
axs[1,0].set_ylabel('BERTScore')
axs[1,0].set_title('Simple Seq2Seq – BERTScore')
axs[1,0].grid(True)

axs[1,1].plot(epochs, cosine, 'o-', color='red')
axs[1,1].set_xlabel('Epoch')
axs[1,1].set_ylabel('Cosine Similarity')
axs[1,1].set_title('Simple Seq2Seq – Cosine Similarity')
axs[1,1].grid(True)

plt.tight_layout()
plt.savefig('simple_seq2seq_training_metrics.png', dpi=150)
plt.show()