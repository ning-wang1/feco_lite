ben_nums = [49548, 39100, 13113, 175240, 62154, 99514, 46585, 19528, 52150]
mirai_nums = [652100, 0, 512133, 610714, 436010, 429337, 513248, 514860, 0]
bashlite_num = [316650, 316400, 310630, 312723, 330096, 309040, 303223, 316438, 323072]

ben_sum = sum(ben_nums)
mirai_sum = sum(mirai_nums)
bashlite_sum = sum(bashlite_num)

total = ben_sum + mirai_sum +bashlite_sum

print(f'the total number of benign samples {ben_sum}')
print(f'the total number of mirai samples {mirai_sum}')
print(f'the total number of bashlite samples {bashlite_sum}')
print(f'the total number of malicious samples {bashlite_sum + mirai_sum}')
print(f'the total number of all samples {total}')