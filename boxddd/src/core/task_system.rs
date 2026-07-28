use crate::core::callback_state::CallbackGuard;
use crate::error::Error;
use boxddd_sys::ffi;
use std::os::raw::{c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Condvar, Mutex, OnceLock, mpsc};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

/// Safe task-system adapter installed on a Box3D world definition.
///
/// Box3D calls `enqueueTask` during `b3World_Step` and later calls `finishTask`
/// for every non-null task handle returned by enqueue. `finishTask` must block
/// until that task has completed; schedulers that cannot provide this blocking
/// guarantee must not be installed through this adapter. Do not call
/// `World::try_step` from a job system worker that cannot park or join child
/// work; that scheduler shape can deadlock when Box3D asks `finishTask` to wait.
#[derive(Clone, Debug)]
pub struct TaskSystem {
    inner: Arc<TaskSystemInner>,
}

impl TaskSystem {
    /// Runs Box3D tasks on a lazily initialized, process-wide pool of blocking
    /// operating-system threads.
    ///
    /// The pool is sized to [`std::thread::available_parallelism`] and reuses
    /// its workers across tasks, steps, and worlds. This scheduler contains
    /// panics, returns them as [`Error::CallbackPanicked`] from
    /// `World::try_step`, and blocks each corresponding Box3D `finishTask`
    /// callback until that task has completed.
    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn blocking_threads() -> Self {
        Self::new(FaultMode::None)
    }

    /// Tries to create the blocking-thread scheduler.
    ///
    /// Browser and WASI targets do not expose this scheduler because Box3D requires
    /// `finishTask` to block until child tasks complete.
    #[inline]
    pub fn try_blocking_threads() -> crate::Result<Self> {
        #[cfg(target_arch = "wasm32")]
        {
            Err(Error::UnsupportedOnWasm)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(Self::blocking_threads())
        }
    }

    /// Returns counters for diagnostics and tests.
    #[inline]
    pub fn stats(&self) -> TaskSystemStats {
        self.inner.stats()
    }

    #[doc(hidden)]
    #[inline]
    pub fn __guard_rejections_for_test(&self) -> usize {
        self.inner.guard_rejections.load(Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn raw_context(&self) -> *mut c_void {
        Arc::as_ptr(&self.inner) as *mut c_void
    }

    #[inline]
    pub(crate) fn reset_panics(&self) {
        self.inner.panicked.store(false, Ordering::Release);
    }

    #[inline]
    pub(crate) fn panicked(&self) -> bool {
        self.inner.panicked.load(Ordering::Acquire)
    }

    #[inline]
    fn new(fault_mode: FaultMode) -> Self {
        Self {
            inner: Arc::new(TaskSystemInner::new(fault_mode)),
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn __panic_on_enqueue_for_test() -> Self {
        Self::new(FaultMode::PanicOnEnqueue)
    }

    #[doc(hidden)]
    #[inline]
    pub fn __panic_on_task_for_test() -> Self {
        Self::new(FaultMode::PanicOnTask)
    }

    #[doc(hidden)]
    #[inline]
    pub fn __panic_on_finish_for_test() -> Self {
        Self::new(FaultMode::PanicOnFinish)
    }

    #[doc(hidden)]
    #[inline]
    pub fn __check_callback_guard_for_test() -> Self {
        Self::new(FaultMode::CheckCallbackGuard)
    }
}

/// Snapshot of a [`TaskSystem`]'s task counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskSystemStats {
    /// Number of tasks enqueued.
    pub enqueued: usize,
    /// Number of tasks started.
    pub started: usize,
    /// Number of tasks completed.
    pub completed: usize,
    /// Number of `finishTask` callbacks observed.
    pub finished: usize,
    /// Number of task callbacks that panicked.
    pub panicked: bool,
}

#[derive(Debug)]
struct TaskSystemInner {
    fault_mode: FaultMode,
    panicked: AtomicBool,
    enqueued: AtomicUsize,
    started: AtomicUsize,
    completed: AtomicUsize,
    finished: AtomicUsize,
    guard_rejections: AtomicUsize,
}

impl TaskSystemInner {
    fn new(fault_mode: FaultMode) -> Self {
        Self {
            fault_mode,
            panicked: AtomicBool::new(false),
            enqueued: AtomicUsize::new(0),
            started: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            guard_rejections: AtomicUsize::new(0),
        }
    }

    fn stats(&self) -> TaskSystemStats {
        TaskSystemStats {
            enqueued: self.enqueued.load(Ordering::Relaxed),
            started: self.started.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            finished: self.finished.load(Ordering::Relaxed),
            panicked: self.panicked.load(Ordering::Acquire),
        }
    }

    fn mark_panicked(&self) {
        self.panicked.store(true, Ordering::Release);
    }

    fn run_task(&self, invocation: TaskInvocation) {
        self.started.fetch_add(1, Ordering::Relaxed);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = CallbackGuard::enter();
            if self.fault_mode == FaultMode::CheckCallbackGuard
                && matches!(
                    crate::core::callback_state::check_not_in_callback(),
                    Err(Error::InCallback)
                )
            {
                self.guard_rejections.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(task) = invocation.task {
                unsafe { task(invocation.task_context) };
            }
            if self.fault_mode == FaultMode::PanicOnTask {
                panic!("injected Box3D task panic");
            }
        }));
        if result.is_err() {
            self.mark_panicked();
        }
        self.completed.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultMode {
    #[cfg(not(target_arch = "wasm32"))]
    None,
    PanicOnEnqueue,
    PanicOnTask,
    PanicOnFinish,
    CheckCallbackGuard,
}

#[derive(Clone, Copy)]
struct TaskInvocation {
    task: ffi::b3TaskCallback,
    task_context: *mut c_void,
}

unsafe impl Send for TaskInvocation {}

#[cfg(not(target_arch = "wasm32"))]
struct TaskHandle {
    completion: Arc<TaskCompletion>,
}

#[cfg(not(target_arch = "wasm32"))]
struct TaskCompletion {
    done: Mutex<bool>,
    ready: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
impl TaskCompletion {
    fn new() -> Self {
        Self {
            done: Mutex::new(false),
            ready: Condvar::new(),
        }
    }

    fn complete(&self) {
        *self.done.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.ready.notify_one();
    }

    fn wait(&self) {
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = self.ready.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct PoolJob {
    scheduler: Arc<TaskSystemInner>,
    invocation: TaskInvocation,
    completion: Arc<TaskCompletion>,
}

#[cfg(not(target_arch = "wasm32"))]
struct BlockingPool {
    sender: mpsc::Sender<PoolJob>,
}

#[cfg(not(target_arch = "wasm32"))]
fn blocking_pool() -> &'static BlockingPool {
    static POOL: OnceLock<BlockingPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<PoolJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        let threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .max(1);
        for worker in 0..threads {
            let receiver = receiver.clone();
            thread::Builder::new()
                .name(format!("boxddd-worker-{worker}"))
                .spawn(move || {
                    loop {
                        let job = {
                            let recv = receiver.lock().unwrap_or_else(|e| e.into_inner());
                            recv.recv()
                        };
                        let Ok(job) = job else {
                            break;
                        };
                        job.scheduler.run_task(job.invocation);
                        job.completion.complete();
                    }
                })
                .expect("failed to spawn BoxDDD task worker");
        }
        BlockingPool { sender }
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) unsafe extern "C" fn enqueue_task(
    task: ffi::b3TaskCallback,
    task_context: *mut c_void,
    user_context: *mut c_void,
    _task_name: *const c_char,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let scheduler = unsafe { scheduler_from_context(user_context) };
        scheduler.enqueued.fetch_add(1, Ordering::Relaxed);
        if task.is_none() {
            return std::ptr::null_mut();
        }

        let invocation = TaskInvocation { task, task_context };
        if scheduler.fault_mode == FaultMode::PanicOnEnqueue {
            scheduler.run_task(invocation);
            panic!("injected Box3D enqueue panic");
        }

        let scheduler_for_task = unsafe { clone_scheduler(scheduler as *const TaskSystemInner) };
        let completion = Arc::new(TaskCompletion::new());
        let job = PoolJob {
            scheduler: scheduler_for_task,
            invocation,
            completion: completion.clone(),
        };
        if blocking_pool().sender.send(job).is_err() {
            scheduler.run_task(invocation);
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(TaskHandle { completion })).cast()
    }));

    match result {
        Ok(user_task) => user_task,
        Err(_) => {
            if let Some(scheduler) = unsafe { scheduler_from_context_checked(user_context) } {
                scheduler.mark_panicked();
            }
            std::ptr::null_mut()
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) unsafe extern "C" fn enqueue_task(
    task: ffi::b3TaskCallback,
    task_context: *mut c_void,
    user_context: *mut c_void,
    _task_name: *const c_char,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let scheduler = unsafe { scheduler_from_context(user_context) };
        scheduler.enqueued.fetch_add(1, Ordering::Relaxed);
        if let Some(task) = task {
            scheduler.run_task(TaskInvocation {
                task: Some(task),
                task_context,
            });
        }
    }));

    if result.is_err() {
        if let Some(scheduler) = unsafe { scheduler_from_context_checked(user_context) } {
            scheduler.mark_panicked();
        }
    }
    std::ptr::null_mut()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) unsafe extern "C" fn finish_task(user_task: *mut c_void, user_context: *mut c_void) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if user_task.is_null() {
            return;
        }
        let scheduler = unsafe { scheduler_from_context(user_context) };

        let handle = unsafe { Box::from_raw(user_task.cast::<TaskHandle>()) };
        handle.completion.wait();
        scheduler.finished.fetch_add(1, Ordering::Relaxed);
        if scheduler.fault_mode == FaultMode::PanicOnFinish {
            panic!("injected Box3D finish panic");
        }
    }));

    if result.is_err() {
        if let Some(scheduler) = unsafe { scheduler_from_context_checked(user_context) } {
            scheduler.mark_panicked();
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) unsafe extern "C" fn finish_task(user_task: *mut c_void, user_context: *mut c_void) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let scheduler = unsafe { scheduler_from_context(user_context) };
        if !user_task.is_null() {
            scheduler.mark_panicked();
        }
        scheduler.finished.fetch_add(1, Ordering::Relaxed);
    }));

    if result.is_err() {
        if let Some(scheduler) = unsafe { scheduler_from_context_checked(user_context) } {
            scheduler.mark_panicked();
        }
    }
}

#[inline]
pub(crate) fn install_callbacks(raw_def: &mut ffi::b3WorldDef, task_system: &TaskSystem) {
    raw_def.enqueueTask = Some(enqueue_task);
    raw_def.finishTask = Some(finish_task);
    raw_def.userTaskContext = task_system.raw_context();
}

unsafe fn scheduler_from_context<'a>(context: *mut c_void) -> &'a TaskSystemInner {
    debug_assert!(!context.is_null());
    unsafe { &*context.cast::<TaskSystemInner>() }
}

unsafe fn scheduler_from_context_checked<'a>(context: *mut c_void) -> Option<&'a TaskSystemInner> {
    if context.is_null() {
        None
    } else {
        Some(unsafe { scheduler_from_context(context) })
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn clone_scheduler(ptr: *const TaskSystemInner) -> Arc<TaskSystemInner> {
    unsafe {
        Arc::increment_strong_count(ptr);
        Arc::from_raw(ptr)
    }
}
