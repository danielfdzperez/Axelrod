# -*- coding: utf-8 -*-


def sparkline(actions):
    return u''.join([u'█' if play == 'C' else u' ' for play in actions])


class Match(object):

    def __init__(self, players, turns, deterministic_cache=None,
                 cache_mutable=True, noise=0):
        """
        Parameters
        ----------
        players : tuple
            A pair of axelrod.Player objects
        turns : integer
                The number of turns per match
        deterministic_cache : dictionary
            A cache of resulting actions for stochastic matches
        cache_mutable : boolean
            Whether the deterministic cache can be updated or not
        noise : float
            The probability that a player's intended action should be flipped
        """
        self._player1 = players[0]
        self._player2 = players[1]
        self._classes = (players[0].__class__, players[1].__class__)
        self._turns = turns
        if deterministic_cache is None:
            self._cache = {}
        else:
            self._cache = deterministic_cache
        self._cache_mutable = cache_mutable
        self._noise = noise
        self._result = self._generate_result()

    @property
    def _stochastic(self):
        """
        A boolean to show whether a match between two players would be
        stochastic
        """
        return (
            self._noise or
            self._player1.classifier['stochastic'] or
            self._player2.classifier['stochastic'])

    @property
    def _cache_update_required(self):
        """
        A boolean to show whether the determinstic cache should be updated
        """
        return (
            not self._noise and
            self._cache_mutable and not (
                self._player1.classifier['stochastic']
                or self._player2.classifier['stochastic'])
        )

    @property
    def result(self):
        return self._result

    def _generate_result(self):
        """
        The resulting list of actions from a match between two players.

        This function determines whether the actions list can be obtained from
        the deterministic cache and returns it from there if so. If not, it
        calls the play method and returns the list from there.

        This is implemented as an ordinary method rather than a property setter
        because we aren't passing in a value.

        Returns
        -------
        A list of the form:

        e.g. for a 2 turn match between Cooperator and Defector:

            [(C, C), (C, D)]

        i.e. One entry per turn containing a pair of actions.
        """
        if (self._stochastic or self._classes not in self._cache):
            return self._play()
        else:
            return self._cache[self._classes]

    def _play(self):
        """
        Plays the match and returns the resulting list of actions

        This function is called by the results method if the deterministic
        cache cannot be used.

        Returns
        -------
        A list of the form:

        e.g. for a 2 turn match between Cooperator and Defector:

            [(C, C), (C, D)]

        i.e. One entry per turn containing a pair of actions.
        """
        turn = 0
        self._player1.reset()
        self._player2.reset()
        while turn < self._turns:
            turn += 1
            self._player1.play(self._player2, self._noise)
        result = list(zip(self._player1.history, self._player2.history))
        if self._cache_update_required:
            self._cache[self._classes] = result
        return result

    @property
    def sparklines(self):
        return (
            sparkline(self._player1.history) +
            u'\n' +
            sparkline(self._player2.history))