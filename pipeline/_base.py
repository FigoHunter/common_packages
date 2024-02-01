import traceback

class Job:
    def __init__(self, name, func):
        self._name = name
        self._func = func
        self.validate = None
    
    @property
    def name(self):
        return self._name
    
    @property
    def func(self):
        return self._func

    def _validate(self):
        try:
            return self.validate is not None and self.validate()
        except Exception as e:
            print(traceback.print_exception(e))
            return False

    def _run(self):
        try:
            self.func()
        except Exception as e:
            print(f'Job Exec Failed: {e}')
            raise e

            


class PipelineManager:
    def __init__(self) -> None:
        self._jobs={}
        self._job_dep={}

    def reg_job(self, name, dependencies=[]):
        def decorator(func):
            self._reg_job(name, func, dependencies)
            return func
        return decorator


    def reg_job_validate(self, name):
        def decorator(func):
            self._reg_job_validate(name, func)
            return func
        return decorator

    def run_job(self, name):
        self._run_job(name)

    def _reg_job(self, name, func, dependencies):
        for dep in dependencies:
            if dep not in self._jobs:
                raise Exception(f'Dependencies not found: {dep}')
        self._check_dep_loop(name, dependencies)
        self._jobs[name] = Job(name, func)
        self._reg_job_dep(name, dependencies)

    def _check_dep_loop(self, name, dependencies, list=[]):
        if name in list:
            raise Exception(f'Loop Found: {list}')
        list.append(name)
        for dep in dependencies:
            self._check_dep_loop(dep, self.get_job_dependencies(dep), list)

    def _reg_job_dep(self, name, dependencies):
        self._job_dep[name] = dependencies

    def get_job_dependencies(self, name):
        return self._job_dep.get(name, [])

    def _reg_job_validate(self, name, func):
        if name not in self._jobs:
            raise Exception(f'Job not found: {name}')
        job:Job = self._jobs[name]
        job.validate = func

    def _run_job(self, name):
        job:Job = self._jobs[name]
        flag = False
        for dep in self.get_job_dependencies(name):
            f, _ = self._run_job(dep)
            flag = flag or f
        if flag or not job._validate():
            job._run()
            return True
        else:
            return False

        