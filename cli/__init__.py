"""Administrative CLI for stigmergic pheromone state checkpointing.

Commands
--------
* ``export``  — dump the latest checkpoint (from Redis or disk) to a JSON file.
* ``import``  — load a JSON snapshot and persist it as the latest checkpoint
  so the next router boot warm-starts from it.
* ``inspect`` — print a human-readable summary of a snapshot file.
"""
