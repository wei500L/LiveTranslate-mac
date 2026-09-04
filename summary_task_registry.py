"""Application-level registry for AI-summary worker threads.

The records page used to manage its one ``SummaryWorker`` alone: on page
teardown it re-parented the still-running QThread to the QApplication and
hoped for the best. That left three holes — a cancelled-but-draining thread
had only Qt's parent chain as a strong reference, the app exit had no single
place to request cancellation, and a page destroyed and re-created could not
tell an old worker's late ``finished`` from the new one's.

This module is that single place. It is plain Python (no Qt widget parents,
no signal wiring); the page and the app both talk to it:

* ``register(worker)``      — takes a strong reference and hooks finished;
* ``cancel_all()``          — app exit: cooperative cancel, no terminate(),
                               no long wait on the GUI thread;
* ``prune()``               — drop finished workers (called from finished);
* ``active()``              — introspection for tests and status text.

Session/generation validation stays on the worker (``session``,
``generation``) and on the page; the registry never routes results — it only
guarantees the threads are referenced, cancellable and cleaned up exactly
once.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("LiveTranslate.SummaryTasks")


class SummaryTaskRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._workers = {}  # id(worker) -> worker, insertion-ordered
        self._cancelled = False

    # --- registration -----------------------------------------------------

    def register(self, worker) -> None:
        """Take full ownership of the worker: strong reference *and* Qt parent.

        The page creates the worker with ``parent=page``; a QThread deleted
        with a parent while ``run()`` is still executing is the classic
        "Destroyed while thread is still running" abort. So registration
        detaches it from the page and holds it here: the registry (owned by
        the app) outlives the page, the reference keeps the Python wrapper
        alive, and cleanup happens exactly once — ``prune()``/finished calls
        ``deleteLater()`` on the GUI thread after ``run()`` returned.
        """
        if worker is None:
            return
        with self._lock:
            if id(worker) in self._workers:
                return
            self._workers[id(worker)] = worker
        try:
            # Detach from the page: the registry is the sole owner now.
            worker.setParent(None)
        except Exception:
            log.debug("Could not detach worker from its page", exc_info=True)
        try:
            # finished is delivered on the GUI thread after run() returns;
            # that is the documented-safe point to free the QThread object.
            # A worker registered before start() emits it when its run ends;
            # one that never runs never emits — but the page always starts
            # what it registers, and cancel_all/prune bound the lifetime.
            worker.finished.connect(
                lambda w=worker: self._on_worker_finished_signal(w)
            )
        except Exception:
            log.debug("Could not hook worker finished", exc_info=True)

    def _on_worker_finished_signal(self, worker) -> None:
        """finished hook: prune the registry, free the QThread once.

        deleteLater() from a finished signal is the documented-safe way to
        free a QThread object (run() has returned by the time finished is
        delivered on the GUI thread).
        """
        if worker is None:
            return
        with self._lock:
            self._workers.pop(id(worker), None)
        try:
            worker.deleteLater()
        except Exception:
            pass

    def prune_worker(self, worker) -> None:
        """Remove one worker from the registry (its run() ended)."""
        if worker is None:
            return
        with self._lock:
            self._workers.pop(id(worker), None)

    def prune(self) -> None:
        """Drop references to workers whose run() has ended.

        Only the reference — deleteLater() is exclusively driven by the
        finished signal (delivered on the GUI thread after run() returned).
        The test is ``isFinished()``, *not* ``not isRunning()``: a worker
        registered but not yet started is neither running nor finished, and
        dropping it here would leave a just-started thread with no owner if
        the page is destroyed moments later — the exact
        "destroyed while running" abort this registry exists to prevent.
        isRunning() would also race the run-to-finished transition
        (isRunning stays True briefly after run() returns).
        """
        with self._lock:
            dead = [w for w in self._workers.values() if w.isFinished()]
            for w in dead:
                self._workers.pop(id(w), None)

    # --- cancellation -----------------------------------------------------

    def cancel_all(self) -> None:
        """Request cooperative cancellation of every live worker.

        App-exit path: no ``terminate()``, no ``wait()`` on the GUI thread —
        each worker's ``cancel()`` sets its flag *and closes the underlying
        OpenAI/httpx client*, so a synchronous request already on the wire is
        interrupted by the closed transport rather than waiting out the
        120s per-request timeout. Honest limits: a request that is mid-read
        on a genuinely hung socket can still take a moment to surface the
        error; closing the transport is an interrupt, not a kill.
        """
        with self._lock:
            self._cancelled = True
            workers = list(self._workers.values())
        for worker in workers:
            try:
                worker.cancel()
            except Exception:
                log.debug("Cancel failed for a summary worker", exc_info=True)

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def active(self) -> list:
        """Live workers, snapshot copy (for status text and tests)."""
        self.prune()
        with self._lock:
            return list(self._workers.values())

    def worker_for(self, session: str):
        """The live worker registered for a session, if any."""
        for worker in self.active():
            if getattr(worker, "session", None) == session:
                return worker
        return None
