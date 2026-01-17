# Deep Q-Network (DQN) Stock Trading Agent

A reinforcement learning-based stock trading system using Deep Q-Networks to learn optimal buy/sell strategies with risk management guardrails.

## Project Overview

This project implements an autonomous stock trading agent using Deep Q-Learning (DQN), a model-free reinforcement learning algorithm that learns to make profitable trading decisions by interacting with historical market data. The agent learns when to buy, hold, or sell stocks to maximize cumulative profit while managing risk through configurable guardrails (stop-loss and take-profit thresholds).

**Key Features:**
- Deep Q-Network with experience replay for stable learning
- Action masking to prevent invalid trades (e.g., selling without inventory)
- Configurable risk management (stop-loss, take-profit, position limits)
- Bollinger Bands as technical indicators for state representation
- Rolling normalization for feature scaling

---

## Data Preprocessing

### Data Source
- **Symbol**: GOOG (Google) - 6-month historical data (2009-2010)
- **Frequency**: Daily (1-day bars)
- **Features**: Open, High, Low, Close, Adjusted Close, Volume

### Preprocessing Steps

1. **Missing Data Handling**
   - Forward fill (`ffill`) to propagate last valid observation forward
   - Ensures no NaN values in the dataset

2. **Feature Selection**
   - **Close Price**: Primary price signal
   - **Bollinger Bands (20-day)**:
     - Upper Band: `MA20 + 2 × STD20`
     - Lower Band: `MA20 - 2 × STD20`
   - These technical indicators capture volatility and price trends

3. **Normalization**
   - **Method**: StandardScaler (Z-score normalization)
   - Centers data to mean=0, scales to unit variance
   - Applied to: Close, BB_upper, BB_lower
   - Normalizers stored for inverse transformation during trading

4. **Train/Test Split**
   - **Training Set**: 50% (first half of data)
   - **Test Set**: 50% (second half of data)
   - No temporal overlap to prevent look-ahead bias

---

## Feature Engineering

### State Representation

The agent observes the market through a **windowed state representation** using sigmoid-transformed price differences:

**Window Size**: 1 day (configurable)

**State Calculation** (for window size `n`):
```
For each time step t:
  - Collect last n+1 data points: [t-n, ..., t-1, t]
  - For each feature (Close, BB_upper, BB_lower):
      - Calculate difference: diff = data[i+1] - data[i]
      - Apply sigmoid: sigmoid(diff) = 1 / (1 + e^(-diff))
  - Flatten into state vector
```

**State Size**: `window_size × num_features = 1 × 3 = 3`

**Why Sigmoid?**
- Normalizes price changes to [0, 1] range
- Emphasizes relative price movements over absolute values
- Helps neural network learn patterns more effectively

---

## DQN Architecture

### Neural Network Structure

```
Input Layer:    state_size = 3 (1 window × 3 features)
                     ↓
Hidden Layer 1: 64 neurons, ReLU activation
                     ↓
Hidden Layer 2: 32 neurons, ReLU activation
                     ↓
Hidden Layer 3: 8 neurons, ReLU activation
                     ↓
Output Layer:   3 neurons (Q-values for each action), Linear activation
```

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Loss Function | Mean Squared Error (MSE) | Measures difference between predicted and target Q-values |
| Optimizer | Adam | Learning rate: 0.001 |
| Batch Size | 32 | Number of experiences sampled per training step |
| Replay Buffer | 1000 | Maximum number of experiences stored |
| Gamma (γ) | 0.95 | Discount factor for future rewards |
| Epsilon (ε) | 1.0 → 0.01 | Exploration rate (decays by 0.99 each episode) |
| Episodes | 2 (configurable) | Number of complete passes through training data |

### Learning Algorithm

**Q-Learning Update Rule** (Bellman Equation):
```
Q(s, a) ← Q(s, a) + α [r + γ × max Q(s', a') - Q(s, a)]
                              a'
```

**Experience Replay**:
- Stores transitions: `(state, action, reward, next_state, done)`
- Samples random mini-batches to break temporal correlation
- Prevents catastrophic forgetting
- Improves sample efficiency

---

## Trading Strategy

### Action Space

The agent has **3 discrete actions**:

| Action | ID | Description |
|--------|-----|-------------|
| **HOLD** | 0 | Take no action (maintain current position or stay in cash) |
| **BUY** | 1 | Purchase `MAX_SHARES` shares at current price |
| **SELL** | 2 | Sell all shares in current position at current price |

**Action Masking**:
- **SELL action is masked** when no position exists (inventory = 0)
- Prevents invalid trades and unnecessary negative rewards
- During exploration (ε-greedy), agent only samples from valid actions

### Reward Structure

| Scenario | Reward | Purpose |
|----------|--------|---------|
| **Buy** | 0 | No immediate reward (neutral action) |
| **Sell** | `(sell_price - buy_price) × num_shares` | Realized profit or loss |
| **Hold (no position)** | `-HOLDING_PENALTY` (-0.05) | Encourages entering positions |
| **Hold (with position)** | 0 | Allows agent to wait for optimal exit |
| **Stop Loss Trigger** | `(current_price - buy_price) × num_shares` | Negative reward (enforced exit) |
| **Take Profit Trigger** | `(current_price - buy_price) × num_shares` | Positive reward (enforced exit) |

### Position Management

- **MAX_SHARES**: 1 (currently always buy/sell this amount)
- **Single Position**: Agent can hold at most 1 position at a time
- **FIFO (First-In-First-Out)**: Oldest shares sold first (not applicable with single position)
- **No Same-Day Round Trip**: Cannot buy on the same day as selling

---

## Guardrails

Risk management rules that override agent actions to prevent excessive losses:

### 1. Stop Loss (30%)
```python
if (current_price - buy_price) / buy_price <= -0.30:
    # Force sell all shares
    # Record loss as negative reward
```
**Purpose**: Limit maximum loss per trade to 30%

### 2. Take Profit (20%)
```python
if (current_price - buy_price) / buy_price >= 0.20:
    # Force sell all shares
    # Record profit as positive reward
```
**Purpose**: Lock in profits at 20% gain

### 3. Max Position Limit
```python
if action == BUY and position is not None:
    action = HOLD  # Convert to hold
```
**Purpose**: Prevent multiple concurrent positions (currently limited to 1)

### 4. No Same-Day Round Trip
```python
if action == BUY and sold_today == True:
    action = HOLD  # Convert to hold
```
**Purpose**: Prevent pattern day trading violations and overtrading

---

## State Machine

### Trading States

```
┌─────────────┐
│   NO        │
│  POSITION   │  ← Initial State
│ (Cash Only) │
└──────┬──────┘
       │
       │ BUY action
       ↓
┌─────────────┐
│  HOLDING    │
│  POSITION   │
│ (Inventory) │
└──────┬──────┘
       │
       │ SELL action / Stop Loss / Take Profit
       ↓
┌─────────────┐
│   NO        │
│  POSITION   │
│ (Cash Only) │
└─────────────┘
```

### State Transitions

| Current State | Action | Next State | Reward |
|--------------|--------|------------|--------|
| No Position | HOLD | No Position | `-HOLDING_PENALTY` |
| No Position | BUY | Holding Position | 0 |
| No Position | SELL | No Position | 0 (masked - invalid) |
| Holding Position | HOLD | Holding Position | 0 |
| Holding Position | BUY | Holding Position | 0 (guardrail converts to HOLD) |
| Holding Position | SELL | No Position | `profit/loss × num_shares` |

### Position Tracking

**Data Structure**:
```python
position = None                      # No position
position = (buy_price, num_shares)   # Active position
```

**Example**:
```python
# Buy 1 share at $10.50
position = (10.50, 1)

# Later sell at $11.00
profit = (11.00 - 10.50) × 1 = $0.50
position = None
```

---

## Training Configuration

### Hyperparameters

```python
# Data Configuration
window_size = 1              # Days of historical data in state
train_test_split = 0.5       # 50% train, 50% test

# Network Architecture
state_size = 3               # window_size × num_features
action_size = 3              # Hold, Buy, Sell
hidden_layers = [64, 32, 8]  # Neurons per layer

# Training Parameters
episode_count = 2            # Training episodes
batch_size = 32              # Experience replay batch size
replay_buffer_size = 1000    # Max experiences stored
learning_rate = 0.001        # Adam optimizer learning rate

# RL Parameters
gamma = 0.95                 # Discount factor
epsilon_start = 1.0          # Initial exploration rate
epsilon_min = 0.01           # Minimum exploration rate
epsilon_decay = 0.99         # Decay per episode

# Guardrails
MAX_SHARES = 1               # Shares per transaction
STOP_LOSS_PCT = 0.30         # 30% stop loss
TAKE_PROFIT_PCT = 0.20       # 20% take profit
HOLDING_PENALTY = 0.05       # Penalty for holding cash
```

### Training Process

1. **Initialize** agent with random weights
2. For each **episode** (1 to `episode_count`):
   - Reset position and profit counters
   - For each **time step** in training data:
     - Get current state (windowed features)
     - **Select action** using ε-greedy policy with action masking
     - **Apply guardrails** (stop loss, take profit, position limits)
     - **Execute action** and calculate reward
     - Store experience: `(state, action, reward, next_state, done)`
     - **Experience replay**: Sample batch and train network
     - Update epsilon (exploration decay)
   - **Save model** after each episode
   - Plot trading behavior and training loss
3. Load best model (last episode) for testing

---

## Test Phase

### Test Setup

```python
# Load trained model
model_filename = 'model_ep1.keras'  # Last episode
agent = Agent(test_mode=True, model_name=model_filename)

# Test on second half of data (unseen during training)
test_data = X_test  # 50% of total dataset
```

### Test Execution

1. **Load trained model** (no exploration, ε = 0)
2. **Initialize** empty position
3. For each time step in test data:
   - Observe state
   - **Predict action** (greedy: argmax Q-values with action masking)
   - Apply guardrails
   - Execute action
   - Track cumulative profit
4. **Report final metrics**

### Sample Test Output

```
Loading model from last episode: model_ep1.keras

Buy: 1 shares at $11.20 (Total: $11.20)
Sell: 1 shares at $11.45 (Total: $11.45) | Profit: $0.25
Buy: 1 shares at $10.95 (Total: $10.95)
TAKE PROFIT triggered: Sold 1 shares at $13.14 (bought at $10.95) | Profit: $2.19
Buy: 1 shares at $12.80 (Total: $12.80)
Sell: 1 shares at $13.05 (Total: $13.05) | Profit: $0.25

------------------------------------------
Total Profit: $2.69
Total Winners: $2.69
Total Losers: $0.00
------------------------------------------
```

### Performance Metrics

- **Total Profit**: Net profit/loss across all trades
- **Total Winners**: Sum of profitable trades
- **Total Losers**: Sum of losing trades
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Buy-sell pairs executed

### Visualization

The test phase generates a trading chart showing:
- **Black line**: Stock close price
- **Blue lines**: Bollinger Bands (upper/lower)
- **Red triangles (^)**: Buy signals
- **Green triangles (v)**: Sell signals
- **Title**: Total gains from trading

---

## Dependencies

```
python >= 3.8
keras >= 2.12
tensorflow >= 2.12
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
scikit-learn >= 1.0
tqdm >= 4.62
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/udacity-dqn-trading.git
cd udacity-dqn-trading/Project

# Install dependencies
pip install keras tensorflow numpy pandas matplotlib scikit-learn tqdm

# Open Jupyter notebook
jupyter notebook project_nb.ipynb
```

## Usage

1. **Run Data Preprocessing**: Execute cells 1-6 (data loading, cleaning, feature engineering)
2. **Define Agent**: Execute cells 7 (DQN model and Agent class)
3. **Configure Training**: Modify guardrails and hyperparameters in training setup cell
4. **Train Agent**: Run training loop (saves models after each episode)
5. **Test Agent**: Run test cells to evaluate performance on unseen data

---

## Project Structure

```
udacity-dqn-trading/
├── Project/
│   ├── project_nb.ipynb          # Main trading notebook
│   ├── GOOG_2009-2010_6m_RAW_1d.csv  # Historical price data
│   ├── model_ep0.keras            # Trained model (episode 0)
│   └── model_ep1.keras            # Trained model (episode 1)
├── Exercises/                     # Course exercises
├── starter/                       # Starter code
├── README.md                      # This file
└── LICENSE.txt
```

---

## Future Enhancements

- **Variable Buy Quantities**: Expand action space to buy 1-N shares (requires larger action space)
- **Multi-Asset Trading**: Trade multiple stocks simultaneously
- **Advanced Features**: RSI, MACD, Volume indicators, VIX
- **Double DQN**: Reduce overestimation bias with separate target network
- **Prioritized Experience Replay**: Sample important experiences more frequently
- **Portfolio Management**: Position sizing, diversification, portfolio-level guardrails
- **Live Trading Integration**: Real-time data feeds and order execution

---

## License

[MIT License](LICENSE.txt)

---

## Acknowledgments

- Udacity Deep Reinforcement Learning Nanodegree