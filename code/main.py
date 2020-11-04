import math
import random
import matplotlib.pyplot

SMALL_SIZE = 40


class Tree:
    def __init__(self, t, f, b, left_ch, right_ch):
        self.t = t
        self.f = f
        self.b = b
        self.left_ch = left_ch
        self.right_ch = right_ch


EMPTY = Tree('C', -1, -1, -1, -1)


def calc_class_cnts(data, k, m):
    res = [0 for _ in range(k)]
    for d in data:
        res[d[m] - 1] += 1
    return res


def calc_entr(cnts, total):
    sum = 0
    for cnt in cnts:
        if cnt > 0:
            p = cnt / total
            sum -= p * math.log(p)
    return sum


def calc_gini(cnts, total):
    sum = 0
    for cnt in cnts:
        sum += (cnt / total) ** 2
    return 1 - sum


def dfs(cur_data, cur_h, m, k, h, tree, mode):
    global nodes_cnt
    v = nodes_cnt
    nodes_cnt += 1
    cnts = calc_class_cnts(cur_data, k, m)
    total = len(cur_data)
    cls_cnts = 0
    mx = 0
    b = 0
    for i in range(k):
        if cnts[i] > 0:
            cls_cnts += 1
        if mx < cnts[i]:
            mx = cnts[i]
            b = i
    if cur_h == h or cls_cnts == 1:
        tree.append(Tree('C', -1, b, -1, -1))
        return v
    best_sum = 1e18
    best_i = 0
    best_j = 0
    fun = calc_entr if total < SMALL_SIZE else calc_gini
    indexes = [i for i in range(m)]
    if mode:
        m1 = int(math.sqrt(m))
        random.shuffle(indexes)
        indexes = indexes[:m1]
    for i in indexes:
        cnts_right = cnts[:]
        cnts_left = [0 for _ in range(k)]
        cur_data_sorted = sorted(cur_data, key=lambda x: x[i])
        for j in range(total - 1):
            cur_obj_cls = cur_data_sorted[j][m] - 1
            cnts_left[cur_obj_cls] += 1
            cnts_right[cur_obj_cls] -= 1
            if cur_data_sorted[j][i] == cur_data_sorted[j + 1][i]:
                continue
            left_size = j + 1
            right_size = total - left_size
            left_ans = fun(cnts_left, left_size)
            right_ans = fun(cnts_right, right_size)
            cur_sum = (left_size * left_ans + right_size * right_ans) / total
            if best_sum > cur_sum:
                best_sum = cur_sum
                best_i = i
                best_j = j
    cur_data_sorted = sorted(cur_data, key=lambda x: x[best_i])
    best_separator_value = (cur_data_sorted[best_j][best_i] + cur_data_sorted[best_j + 1][best_i]) / 2
    tree.append(EMPTY)
    left_ch = dfs(cur_data_sorted[:best_j + 1], cur_h + 1, m, k, h, tree, mode)
    right_ch = dfs(cur_data_sorted[best_j + 1:], cur_h + 1, m, k, h, tree, mode)
    tree[v] = Tree('Q', best_i, best_separator_value, left_ch, right_ch)
    return v


def get_class_on_tree(tree, obj):
    cur_v = 0
    while True:
        if tree[cur_v].t == 'C':
            return tree[cur_v].b + 1
        if obj[tree[cur_v].f] < tree[cur_v].b:
            cur_v = tree[cur_v].left_ch
        else:
            cur_v = tree[cur_v].right_ch


def get_class_on_forest(trees, k, obj):
    cnts = [0 for _ in range(k)]
    for tree in trees:
        cnts[get_class_on_tree(tree, obj) - 1] += 1
    return max(range(k), key=lambda i: cnts[i]) + 1


def gen_objects(objects):
    res = []
    n = len(objects)
    for i in range(n):
        res.append(objects[random.randint(0, n - 1)])
    return res


def find_best_params(filename):
    best_acc = 0
    best_h = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            h = int(lines[i].split()[2])
            acc = float(lines[i + 1].split()[2])
            if best_acc < acc:
                best_acc = acc
                best_h = h
    return [best_acc, best_h]


def add_lead_zero(num):
    return str(num) if num > 9 else "0" + str(num)


def draw_graphic(input_filename, output_filename):
    global MAX_H
    global nodes_cnt
    points = []
    with open(input_filename, 'r') as f:
        objects = []
        m, k = map(int, f.readline().split())
        n = int(f.readline())
        for i in range(n):
            objects.append([int(x) for x in f.readline().split()])
        for h in range(MAX_H):
            res = []
            nodes_cnt = 0
            dfs(objects, 0, m, k, h, res, False)
            cnt = 0
            right = 0
            for obj in objects:
                cnt += 1
                if obj[m] == get_class_on_tree(res, obj):
                    right += 1
            points.append([h, right / cnt])

    for i in range(len(points) - 1):
        matplotlib.pyplot.plot(points[i][0], points[i][1], "go")
        matplotlib.pyplot.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], "k-")
    matplotlib.pyplot.plot(points[len(points) - 1][0], points[len(points) - 1][1], "go")
    matplotlib.pyplot.savefig(output_filename)
    matplotlib.pyplot.clf()


MAX_H = 11
nodes_cnt = 0
# max_h_dataset = 21
# min_h_dataset = 3
# max_h_test_filename = add_lead_zero(max_h_dataset) + "_test.txt"
# max_h_train_filename = add_lead_zero(max_h_dataset) + "_train.txt"
# min_h_test_filename = add_lead_zero(min_h_dataset) + "_test.txt"
# min_h_train_filename = add_lead_zero(min_h_dataset) + "_train.txt"
# draw_graphic(max_h_test_filename, "max_h_test_graphic.png")
# draw_graphic(max_h_train_filename, "max_h_train_graphic.png")
# draw_graphic(min_h_test_filename, "min_h_test_graphic.png")
# draw_graphic(min_h_train_filename, "min_h_train_graphic.png")

# random.seed(1337228)
# TREE_CNT = 50
# for id in range(1, 22):
#     train_filename = (str(id) if id > 9 else "0" + str(id)) + "_train.txt"
#     test_filename = (str(id) if id > 9 else "0" + str(id)) + "_test.txt"
#     train_objects = []
#     test_objects = []
#     with open(str(id) + "_forest_info.txt", 'w') as f:
#         with open(test_filename, 'r') as test_file:
#             m, k = map(int, test_file.readline().split())
#             n = int(test_file.readline())
#             for i in range(n):
#                 test_objects.append([int(x) for x in test_file.readline().split()])
#         with open(train_filename, 'r') as train_file:
#             m, k = map(int, train_file.readline().split())
#             n = int(train_file.readline())
#             for i in range(n):
#                 train_objects.append([int(x) for x in train_file.readline().split()])
#         forest = []
#         for i in range(TREE_CNT):
#             res = []
#             nodes_cnt = 0
#             dfs(gen_objects(train_objects), 0, m, k, 1e18, res, True)
#             print("Generated tree#" + str(i + 1))
#             forest.append(res)
#         cnt = 0
#         right = 0
#         for obj in train_objects:
#             cnt += 1
#             if obj[m] == get_class_on_forest(forest, k, obj):
#                 right += 1
#         f.write("accuracy on train = " + str(right / cnt) + '\n')
#         cnt = 0
#         right = 0
#         for obj in test_objects:
#             cnt += 1
#             if obj[m] == get_class_on_forest(forest, k, obj):
#                 right += 1
#         f.write("accuracy on test = " + str(right / cnt) + '\n')
#         print("Finished id = " + str(id))
# for id in range(1, 22):
#     train_filename = (str(id) if id > 9 else "0" + str(id)) + "_train.txt"
#     test_filename = (str(id) if id > 9 else "0" + str(id)) + "_test.txt"
#     train_objects = []
#     test_objects = []
#     with open(str(id) + "_tree_info.txt", 'w') as f:
#         with open(test_filename, 'r') as test_file:
#             m, k = map(int, test_file.readline().split())
#             n = int(test_file.readline())
#             for i in range(n):
#                 test_objects.append([int(x) for x in test_file.readline().split()])
#         with open(train_filename, 'r') as train_file:
#             m, k = map(int, train_file.readline().split())
#             n = int(train_file.readline())
#             for i in range(n):
#                 train_objects.append([int(x) for x in train_file.readline().split()])
#         for h in range(MAX_H):
#             res = []
#             nodes_cnt = 0
#             dfs(train_objects, 0, m, k, h, res, False)
#             cnt = 0
#             right = 0
#             for obj in test_objects:
#                 cnt += 1
#                 if obj[m] == get_class_on_tree(res, obj):
#                     right += 1
#             f.write("h = " + str(h) + '\n')
#             f.write("accuracy = " + str(right / cnt) + '\n')
#             print("Finished id = " + str(id) + " h = " + str(h))
# results = []
# for id in range(1, 22):
#     results.append(find_best_params(str(id) + "_tree_info.txt"))
# print(results)
