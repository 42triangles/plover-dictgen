#!/usr/bin/env python3

import sys
import string
import asyncio
import time
import json
import multiprocessing
import subprocess
import signal
from unicodedata import combining, normalize


# https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
LATIN = "ä  æ  ǽ  đ ð ƒ ħ ı ł ø ǿ ö  œ  ß  ŧ ü "
ASCII = "ae ae ae d d f h i l o o oe oe ss t ue"
default_outliers = str.maketrans(
    dict(zip(LATIN.split(), ASCII.split()))
)


def remove_diacritics(s, outliers=default_outliers):
    return "".join(
        c
        for c in normalize("NFD", s.lower().translate(outliers))
        if not combining(c)
    )


def status(msg):
    print(f"\r{msg}", file=sys.stderr)


async def timeout_callback(future, *, timeout, callback):
    future = asyncio.ensure_future(future)
    timeout = asyncio.ensure_future(asyncio.sleep(timeout))
    await asyncio.wait(
        [future, timeout],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if timeout.done():
        callback()
    return await future


async def worker(word, arpabet):
    word = word.lower().strip()
    arpabet = arpabet.strip()

    proc = await asyncio.create_subprocess_exec(
        sys.argv[1],
        sys.argv[2],
        arpabet,
        "".join(
            i
            for i in remove_diacritics(word)
            if i in string.ascii_letters
        ),
        "none" if len(sys.argv) < 5 else sys.argv[4],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    long_run = 30

    try:
        start = time.perf_counter()
        stdout, stderr = await timeout_callback(
            proc.communicate(),
            timeout=long_run,
            callback=lambda: status(
                f"time: {word} is taking more than {long_run}s",
            ),
        )
        end = time.perf_counter()

        valid_stdout = stdout and stdout.strip() != b"[]"
        if proc.returncode == 0 and valid_stdout:
            print(json.dumps({
                "word": word,
                "time": end - start,
                "translations": json.loads(stdout),
            }))

            if "cut" in stderr.decode("utf8").lower():
                status(f"LIMT: {word} hit the state limit")
            if end - start >= long_run:
                status(f"TIME: {word} took {end - start}s")
        else:
            secs = end - start
            status(
                f"FAIL: {word} [{arpabet}] failed in {secs}s:",
            )
            print(
                "    " + stderr.decode("utf8")
                .replace("\n", "\n    "),
                file=sys.stderr,
                end="",
            )
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()


lines = 0
with open(sys.argv[3], 'r') as f:
    for i in f:
        lines += 1

cpu_count = multiprocessing.cpu_count()


async def run():
    loop = asyncio.get_running_loop()
    this = asyncio.current_task()
    for i in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(i, this.cancel)

    cpu_count = multiprocessing.cpu_count()

    workers = []

    try:
        line = 0
        with open(sys.argv[3], 'r') as f:
            for i in f:
                line += 1

                if i[0] not in string.ascii_letters:
                    continue

                while len(workers) >= cpu_count:
                    await asyncio.shield(asyncio.wait(
                        workers,
                        return_when=asyncio.FIRST_COMPLETED,
                    ))

                    workers = [i for i in workers if not i.done()]

                print(
                    f"\r{line}/{lines} {line / lines * 100:.1f}%",
                    end="",
                    file=sys.stderr,
                )
                sys.stderr.flush()

                word, arpabet = i.split("  ", 1)
                word = word.split("(", 1)[0]

                if word.endswith(")"):
                    continue

                workers.append(
                    asyncio.create_task(
                        worker(word, arpabet),
                        name=word,
                    )
                )
    except asyncio.CancelledError:
        pass
    finally:
        print(file=sys.stderr)
        outstanding = [
            i.get_name() for i in workers if not i.done()
        ]
        print(f"Outstanding: {outstanding}", file=sys.stderr)
        try:
            await asyncio.wait(workers)
        except asyncio.CancelledError:
            for i in workers:
                i.cancel()

            try:
                await asyncio.wait(workers)
            except asyncio.CancelledError:
                sys.exit(1)


asyncio.run(run())
