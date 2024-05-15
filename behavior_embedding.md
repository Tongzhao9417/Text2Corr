# 目前尝试的behavior embedding
1. gain_S (自我收益)
2. gain_O (行贿者收益)
3. cost (第三方损失, accept条件下有负数数值，reject条件下为0)
4. third_party_gain (第三方收益, accept条件下third_party_gain = 100 - cost; )
5. total (total, 每轮奖金总额, total = self_gain + other_gain)
6. inequality_S_O (自我收益与行贿者收益的不公平程度，gain_O - gain_S)
7. 
