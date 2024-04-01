import functools
import logging
import os
import shlex
import subprocess


@functools.wraps(subprocess.run)
def run(args, **kwargs):
	logger = logging.getLogger('project_pactum.run')
	logger.debug(shlex.join(args))
	p = subprocess.run(args, **kwargs)
	return p


def main(args):
	from project_pactum.core.base import parse, setup_logging

	setup_logging()

	options = parse(args)

	import logging
	logger = logging.getLogger(__name__)
	logger.debug('ok')
