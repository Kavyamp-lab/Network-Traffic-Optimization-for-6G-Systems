"""
AI-Native Intelligent Network Traffic Optimization for 6G Systems
Corrected & Clean Version
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. NETWORK SIMULATION
# ==========================================

class NetworkSimulator:
    def __init__(self, num_nodes=50, time_steps=1000, total_bandwidth=1000):
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.total_bandwidth = total_bandwidth
        
    def visualize_topology(self):
        G = nx.star_graph(self.num_nodes)
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(6,6))
        nx.draw(G, pos, node_size=50)
        plt.title("6G Star Network Topology")
        plt.show()

    def generate_traffic(self):
        t = np.arange(self.time_steps)
        traffic = np.zeros((self.time_steps, self.num_nodes))

        for i in range(self.num_nodes):
            phase = np.random.uniform(0, 2*np.pi)
            base = 10 + 8 * np.sin(2*np.pi*t/200 + phase)
            noise = np.random.normal(0,2,self.time_steps)
            node_traffic = np.clip(base + noise, 0, None)

            # Add random spikes
            spike_idx = np.random.choice(self.time_steps, size=int(self.time_steps*0.05), replace=False)
            node_traffic[spike_idx] += np.random.uniform(20,60,len(spike_idx))

            traffic[:, i] = node_traffic

        return traffic


# ==========================================
# 2. LSTM MODEL
# ==========================================

class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])   # Take last time step
        return out


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


class AIForecaster:
    def __init__(self, num_nodes, seq_length=10):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.lstm = TrafficLSTM(input_size=num_nodes, output_size=num_nodes)
        self.rf = RandomForestRegressor(n_estimators=50)

    def prepare_data(self, traffic):
        scaled = self.scaler.fit_transform(traffic)
        X, y = create_sequences(scaled, self.seq_length)

        split = int(0.8 * len(X))
        return X[:split], X[split:], y[:split], y[split:]

    def train_lstm(self, X, y, epochs=20):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        optimizer = optim.Adam(self.lstm.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.lstm(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def train_rf(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        self.rf.fit(X_flat, y)

    def predict(self, X):
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            lstm_pred = self.lstm(X_t).numpy()

        rf_pred = self.rf.predict(X.reshape(X.shape[0], -1))

        lstm_pred = self.scaler.inverse_transform(lstm_pred)
        rf_pred = self.scaler.inverse_transform(rf_pred)

        return np.maximum(lstm_pred,0), np.maximum(rf_pred,0)


# ==========================================
# 3. RESOURCE ALLOCATION
# ==========================================

class ResourceAllocator:
    def __init__(self, total_bandwidth, num_nodes):
        self.total_bandwidth = total_bandwidth
        self.num_nodes = num_nodes

    def static_allocation(self, steps):
        bw = self.total_bandwidth / self.num_nodes
        return np.full((steps, self.num_nodes), bw)

    def ai_allocation(self, predicted):
        allocation = np.zeros_like(predicted)

        for t in range(predicted.shape[0]):
            total = np.sum(predicted[t])
            if total == 0:
                allocation[t] = self.total_bandwidth / self.num_nodes
            else:
                allocation[t] = (predicted[t] / total) * self.total_bandwidth

        return allocation


# ==========================================
# 4. METRICS
# ==========================================

def calculate_metrics(actual, allocated):
    throughput = np.minimum(actual, allocated)
    packet_loss = np.maximum(0, actual - allocated)

    loss_rate = np.sum(packet_loss) / np.sum(actual) * 100
    latency = 1 + np.mean(np.maximum(0, actual - allocated))

    return np.sum(throughput), loss_rate, latency


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":

    NUM_NODES = 50
    TIME_STEPS = 500
    TOTAL_BW = 1000

    env = NetworkSimulator(NUM_NODES, TIME_STEPS, TOTAL_BW)
    traffic = env.generate_traffic()

    forecaster = AIForecaster(NUM_NODES)
    X_train, X_test, y_train, y_test = forecaster.prepare_data(traffic)

    print("Training LSTM...")
    forecaster.train_lstm(X_train, y_train)

    print("Training RF...")
    forecaster.train_rf(X_train, y_train)

    lstm_pred, rf_pred = forecaster.predict(X_test)

    actual = forecaster.scaler.inverse_transform(y_test)

    allocator = ResourceAllocator(TOTAL_BW, NUM_NODES)

    bw_static = allocator.static_allocation(len(actual))
    bw_lstm = allocator.ai_allocation(lstm_pred)
    bw_rf = allocator.ai_allocation(rf_pred)

    m_static = calculate_metrics(actual, bw_static)
    m_lstm = calculate_metrics(actual, bw_lstm)
    m_rf = calculate_metrics(actual, bw_rf)

    print("\nPerformance Comparison:")
    print("Static  :", m_static)
    print("RF AI   :", m_rf)
    print("LSTM AI :", m_lstm)

    # Plot latency comparison
    plt.bar(["Static","RF","LSTM"], [m_static[2], m_rf[2], m_lstm[2]])
    plt.title("Latency Comparison")
    plt.ylabel("Latency")
    plt.show()