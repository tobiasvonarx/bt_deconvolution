import main
from tqdm import tqdm
losses = []
for _ in tqdm(range(16)):
    test_prop_y, prop_y, test_loss = main.main(50, True)
    losses.append(test_loss)
print(losses)
