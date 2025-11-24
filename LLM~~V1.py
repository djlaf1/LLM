"MIT License - Copyright (c) 2025 djlaf1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
import argparse
import sys
import os
from datetime import datetime

class UltimateConfig:
    vocab_size: int = 50000
    block_size: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    bias: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class UltimateTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class UltimateAI:
    def __init__(self):
        self.config = UltimateConfig()
        self.model = None
        self.optimizer = None
        self.vocab = {}
        self.reverse_vocab = {}
        self.training_losses = []
        self.training_epochs = 0
        self.best_loss = float('inf')
        self.setup_vocab()
        self.setup_model()
        
    def setup_vocab(self):
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', '<BOS>', '<SEP>']
        ai_terms = [
            'transformer', 'attention', 'token', 'embedding', 'neural', 'network',
            'training', 'inference', 'gradient', 'backpropagation', 'optimizer',
            'loss', 'accuracy', 'epoch', 'batch', 'layer', 'activation', 'weights',
            'bias', 'parameters', 'forward', 'backward', 'dropout', 'normalization',
            'convolution', 'pooling', 'recurrent', 'lstm', 'gru', 'autoencoder',
            'gan', 'reinforcement', 'supervised', 'unsupervised', 'clustering',
            'classification', 'regression', 'overfitting', 'regularization'
        ]
        self.vocab = {}
        token_id = 0
        for token in special_tokens:
            self.vocab[token] = token_id
            token_id += 1
        for term in ai_terms:
            self.vocab[term] = token_id
            token_id += 1
        common_words = """
            hello hi hey greetings welcome goodbye bye see later
            what how when where why which who whose whom
            is am are was were be being been have has had do does did
            can could will would should may might must
            and or but so because if then else while until
            the a an this that these those my your his her its our their
            i you he she it we they me him us them
            good bad great awesome amazing interesting fascinating
            please thank thanks sorry okay yes no maybe
            understand explain tell teach show demonstrate
            ai artificial intelligence machine learning deep learning
            computer science technology programming coding
            model algorithm data dataset training testing validation
            performance metric evaluation benchmark
        """.split()
        for word in common_words:
            if word not in self.vocab:
                self.vocab[word] = token_id
                token_id += 1
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def setup_model(self):
        try:
            self.model = UltimateTransformer(self.config)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2)
            )
        except Exception as e:
            print(f"Model setup error: {e}")
            
    def encode_text(self, text):
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return [self.vocab['<SOS>']] + token_ids + [self.vocab['<EOS>']]
    
    def decode_tokens(self, token_ids):
        tokens = []
        for token_id in token_ids:
            if token_id == self.vocab['<EOS>']:
                break
            if token_id not in [self.vocab['<PAD>'], self.vocab['<SOS>']]:
                token = self.reverse_vocab.get(token_id, '<UNK>')
                if token != '<UNK>':
                    tokens.append(token)
        return ' '.join(tokens)
    
    def get_training_data(self):
        return [
            ("what is artificial intelligence", "artificial intelligence is the simulation of human intelligence processes by machines especially computer systems"),
            ("how do neural networks work", "neural networks are computing systems inspired by biological brains that learn patterns from data through layers of interconnected neurons"),
            ("explain machine learning", "machine learning is a subset of ai that enables computers to learn and make decisions from data without being explicitly programmed"),
            ("what are transformers in ai", "transformers are neural network architectures that use attention mechanisms to process sequential data enabling parallel computation and capturing long range dependencies"),
            ("how does attention work", "attention mechanisms allow models to focus on relevant parts of input when making predictions by computing weighted sums of values based on query key compatibility"),
            ("what is self attention", "self attention is a mechanism where each position in a sequence attends to all other positions in the same sequence to compute representations"),
            ("what is gradient descent", "gradient descent is an optimization algorithm that minimizes loss functions by iteratively moving parameters in the direction of steepest descent"),
            ("explain backpropagation", "backpropagation is the algorithm for calculating gradients in neural networks by applying the chain rule backwards from the output to the input"),
            ("what are epochs in training", "an epoch is one complete pass through the entire training dataset during model training"),
            ("what are tokens in nlp", "tokens are the basic units of text that language models process typically words subwords or characters representing the model vocabulary"),
            ("how does tokenization work", "tokenization breaks text into smaller units called tokens which are then converted to numerical ids for processing by neural networks"),
            ("what is subword tokenization", "subword tokenization splits rare words into smaller meaningful units allowing models to handle unknown words and reduce vocabulary size"),
            ("what are embeddings", "embeddings are dense vector representations of discrete objects like words that capture semantic meaning in continuous space"),
            ("explain layer normalization", "layer normalization stabilizes training by normalizing activations across features for each data point reducing internal covariate shift"),
            ("what is dropout", "dropout is a regularization technique that randomly sets activations to zero during training preventing overfitting and improving generalization"),
            ("hello", "hello how can i help you with ai and machine learning today"),
            ("hi", "hi im an advanced ai assistant powered by transformer architecture"),
            ("hey", "hey lets talk about artificial intelligence and neural networks"),
            ("goodbye", "goodbye feel free to ask more about ai and machine learning"),
            ("thanks", "youre welcome im here to explain ai concepts anytime"),
            ("what is pytorch", "pytorch is an open source machine learning framework that provides tensor computation and deep neural networks"),
            ("explain gpu acceleration", "gpu acceleration uses graphics processing units to parallelize neural network computations dramatically speeding up training and inference"),
            ("what is automatic differentiation", "automatic differentiation automatically computes gradients of mathematical functions enabling efficient backpropagation in deep learning"),
        ]
    
    def prepare_training_batch(self):
        training_pairs = self.get_training_data()
        max_length = 0
        
        for input_text, target_text in training_pairs:
            input_ids = self.encode_text(input_text)
            target_ids = self.encode_text(target_text)
            max_length = max(max_length, len(input_ids), len(target_ids))
        
        max_length = min(max_length, self.config.block_size)
        
        inputs = []
        targets = []
        
        for input_text, target_text in training_pairs:
            input_ids = self.encode_text(input_text)[:max_length]
            target_ids = self.encode_text(target_text)[:max_length]
            
            input_ids += [self.vocab['<PAD>']] * (max_length - len(input_ids))
            target_ids += [self.vocab['<PAD>']] * (max_length - len(target_ids))
            
            inputs.append(input_ids)
            targets.append(target_ids)
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def train(self, epochs=100):
        if self.model is None:
            return "Model not initialized!"
            
        try:
            inputs_tensor, targets_tensor = self.prepare_training_batch()
            
            if inputs_tensor.size(1) > self.config.block_size:
                return f"Sequence length {inputs_tensor.size(1)} exceeds block size {self.config.block_size}"
            
            self.model.train()
            losses = []
            
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                
                logits, loss = self.model(inputs_tensor, targets_tensor)
                
                if torch.isnan(loss):
                    return "Training failed: Loss became NaN"
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                losses.append(loss.item())
                self.training_epochs += 1
                
                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
                    self.save_model()
                
                if epoch % 50 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                    
            self.training_losses.extend(losses)
            return f"Training completed! Final loss: {losses[-1]:.4f}, Best loss: {self.best_loss:.4f}"
            
        except Exception as e:
            return f"Training error: {str(e)}"
    
    def generate_response(self, message):
        if self.model is None:
            return self.fallback_response(message)
            
        try:
            self.model.eval()
            
            input_ids = self.encode_text(message)
            if len(input_ids) > self.config.block_size:
                input_ids = input_ids[:self.config.block_size]
                
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_tensor, 
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature
                )
            
            response_ids = generated_ids[0].tolist()
            response = self.decode_tokens(response_ids[len(input_ids):])
            
            if not response.strip():
                return self.fallback_response(message)
                
            return f"AI: {response.capitalize()}"
            
        except Exception as e:
            return self.fallback_response(message)
    
    def fallback_response(self, message):
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "AI: Hello! I'm an advanced AI powered by transformer architecture. How can I help you understand artificial intelligence?"
        elif any(word in message_lower for word in ['ai', 'artificial', 'intelligence']):
            return "AI: Artificial intelligence involves creating systems that can perform tasks requiring human intelligence, like understanding language and recognizing patterns."
        elif any(word in message_lower for word in ['neural', 'network']):
            return "AI: Neural networks are computing systems inspired by biological brains that learn from data through interconnected layers of artificial neurons."
        elif any(word in message_lower for word in ['transformer', 'attention']):
            return "AI: Transformers use attention mechanisms to process sequences, allowing models to focus on relevant information when making predictions."
        elif any(word in message_lower for word in ['token', 'tokenization']):
            return "AI: Tokens are the basic units of text that AI models process. Tokenization converts text into these units for neural network processing."
        elif any(word in message_lower for word in ['train', 'training']):
            return "AI: Training involves showing models examples to adjust their parameters, using algorithms like gradient descent to minimize prediction errors."
        else:
            fallbacks = [
                "AI: I'm processing your query through my transformer architecture. Could you ask about AI fundamentals, neural networks, or machine learning?",
                "AI: My attention mechanisms are focusing on your message. Try asking about transformer architecture or how AI models work!",
                "AI: I'm optimizing my response generation. What would you like to know about artificial intelligence or deep learning?",
                "AI: I'm performing forward propagation on your input. Ask me about tokens, embeddings, or neural network training!",
            ]
            return random.choice(fallbacks)
    
    def save_model(self):
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'vocab': self.vocab,
                'training_losses': self.training_losses,
                'training_epochs': self.training_epochs,
                'best_loss': self.best_loss,
                'config': self.config.__dict__
            }
            torch.save(checkpoint, "ultimate_ai_model.pth")
        except Exception as e:
            print(f"Save error: {e}")
    
    def load_model(self):
        try:
            if not os.path.exists("ultimate_ai_model.pth"):
                return False
                
            checkpoint = torch.load("ultimate_ai_model.pth", map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.vocab = checkpoint['vocab']
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            self.training_losses = checkpoint['training_losses']
            self.training_epochs = checkpoint['training_epochs']
            self.best_loss = checkpoint['best_loss']
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

class CLIChat:
    def __init__(self):
        self.ai = UltimateAI()
        self.conversation_history = []
        
    def print_banner(self):
        print("=" * 70)
        print("AI")
        print("=" * 70)
        print("Model: trabsformer | Embeddings: 768-dim")
        print("Heads: 12 | Layers: 12 | Context: 1024 tokens")
        print("=" * 70)
        print("Type 'train' to train the model")
        print("Type 'quit' or 'exit' to end the chat")
        print("Type 'clear' to clear conversation history")
        print("Type 'info' to show model information")
        print("=" * 70)
        
    def show_model_info(self):
        param_count = sum(p.numel() for p in self.ai.model.parameters()) if self.ai.model else 0
        print("\n" + "=" * 50)
        print("MODEL INFORMATION")
        print("=" * 50)
        print(f"Architecture: GPT-style Transformer")
        print(f"Embedding Dimensions: {self.ai.config.n_embd}")
        print(f"Attention Heads: {self.ai.config.n_head}")
        print(f"Transformer Layers: {self.ai.config.n_layer}")
        print(f"Context Length: {self.ai.config.block_size} tokens")
        print(f"Vocabulary Size: {len(self.ai.vocab)} tokens")
        print(f"Model Parameters: {param_count:,}")
        print(f"Training Epochs: {self.ai.training_epochs}")
        print(f"Best Loss: {self.ai.best_loss:.4f}")
        print("=" * 50)
        
    def clear_chat(self):
        self.conversation_history.clear()
        print("\nConversation history cleared.")
        
    def train_model(self):
        print("\nStarting model training...")
        print("This may take a few moments...")
        result = self.ai.train(epochs=500)
        print(f"\n{result}")
        
    def chat_loop(self):
        self.print_banner()
        
        try:
            if self.ai.load_model():
                print("Loaded pre-trained model successfully!")
            else:
                print("No pre-trained model found. Use 'train' command to train.")
        except:
            print("No pre-trained model found. Use 'train' command to train.")
            
        print("\nYou can start chatting now:\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! Thanks for chatting.")
                    break
                    
                elif user_input.lower() == 'train':
                    self.train_model()
                    continue
                    
                elif user_input.lower() == 'clear':
                    self.clear_chat()
                    continue
                    
                elif user_input.lower() == 'info':
                    self.show_model_info()
                    continue
                    
                self.conversation_history.append(f"You: {user_input}")
                
                response = self.ai.generate_response(user_input)
                print(response)
                
                self.conversation_history.append(response)
                
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

def main():
    parser = argparse.ArgumentParser(description='Ultimate AI Chat with Transformer Architecture')
    parser.add_argument('--train', action='store_true', help='Train the model before chatting')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    
    args = parser.parse_args()
    
    chat = CLIChat()
    
    if args.train:
        chat.train_model()
        
    chat.chat_loop()

if __name__ == "__main__":
    main()
