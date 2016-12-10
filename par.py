import multiprocessing as mp
jobs = [1, 2, 3, 4]
job_list = []


def my_func(params):
    a, b, c = params
    return a + b + c
    return 1 + 3

def main():
    for job in jobs:
        job_list.append((job, 1, 2))

    print(job_list)

    with mp.Pool() as pool:
        func_ret = pool.map(my_func, job_list)

    print(func_ret)

"""Run main"""
if __name__ == '__main__':
    main()