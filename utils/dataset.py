
def dump_dataset_info(dataset):
    print(f'records: {len(dataset)}')

    # x, y = dataset[0]
    # print(f'data shape: {x.shape}')
    # print(f'data label: {y}')

    # count samples for each class
    classes = {}
    for _, y in dataset:
        classes.setdefault(y, 0)
        classes[y] += 1
    print(f'classes: {len(classes)}')


    # classes = sorted(classes.keys(), key=lambda x: x[1], reverse=True)
    classes = sorted(classes.items(), key=lambda x: x[0])
    for key, value in classes:
        print(f'class[{key}] items: {value}')