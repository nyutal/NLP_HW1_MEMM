import multiprocessing as mp
jobs = [1, 2, 3, 4]
job_list = []


def my_func(params):
    a, b, c = params
    return a+b+c, c*a


def main():
    for job in jobs:
        job_list.append((job, 1, 2))

    print(job_list)

    with mp.Pool() as pool:
        func_ret = pool.map(my_func, job_list)
    a,b = zip(*func_ret)
    print(func_ret)
    print(a)
    print(b)

"""Run main"""
if __name__ == '__main__':
    main()