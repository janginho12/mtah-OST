"""Synthetic restart regressions. All coordinates and times are invented."""
import unittest

import numpy as np

from ost import goal_tag


class KickoffReception(unittest.TestCase):
    def setUp(self):
        self.fps = 25
        self.pitch = np.array([100., 60.])
        self.centre = self.pitch / 2
        self.players = np.tile(self.centre + [[-12., 0.], [0., 0.], [8., 0.]], (150, 1, 1))
        self.ball = np.full((150, 2), np.nan)
        self.ball[60:81] = self.centre + np.column_stack((np.arange(21) * .4, np.zeros(21)))
        self.ball[81:100] = self.centre + [8., 0.]
        self.control = np.full(150, -1, int)
        self.control[81:100] = 1
        self.blocks = np.ones(150, int)
        self.row = dict(half=1, conceding_team='A', start_frame=0, end_frame_exclusive=12)
        self.config = goal_tag.Config(minimum_goal_to_center_seconds=1.)

    def result(self, window=(50, 75)):
        visible = np.isfinite(self.ball).all(axis=1)
        features = dict(raw_ball=self.ball, ball_valid=visible, ball_reliability=visible * .85,
                        ball_speed=np.where(visible, 10., np.nan))
        _, edges = goal_tag.motion.motion(self.players, self.blocks, self.fps, self.config)
        return goal_tag.find_centre_reception(self.row, features, self.players, np.array([0, 1, 1]),
                                              self.blocks, self.fps, self.pitch, self.control, edges,
                                              len(self.blocks), self.config, window)

    def test_moving_at_first_sighting_releases_at_control_even_after_tag_end(self):
        event = self.result()
        self.assertEqual(event['center_start_frame'], 60)
        self.assertLess(event['kick_frame'], event['receive_frame'])
        self.assertEqual(event['receive_frame'], 81)
        self.assertEqual((event['kicker_slot'], event['receiver_slot']), (2, 3))

    def test_moving_centre_crossing_is_not_a_new_origin(self):
        self.ball[40:60] = self.centre + np.column_stack((np.arange(-20, 0) * .4, np.zeros(20)))
        self.assertEqual(self.result()['receive_frame'], -1)

    def test_unrelated_control_or_broken_trajectory_cannot_release(self):
        self.control[72] = 0
        self.assertEqual(self.result()['receive_frame'], -1)
        self.control[72] = -1
        self.ball[72] = np.nan
        self.assertEqual(self.result()['receive_frame'], -1)

    def test_no_restart_tag_or_no_centre_contact_cannot_use_moving_origin(self):
        self.assertEqual(self.result(window=None)['receive_frame'], -1)
        self.players[:, 1] += [0., 6.]
        self.assertEqual(self.result()['receive_frame'], -1)

    def test_wrong_or_out_of_window_tag_cannot_locate_restart(self):
        goal = dict(association_start=10, association_end=85, end_frame_exclusive=65)
        tag = dict(code='SET PIECES', type='Kick Off (After Goal)', start=50, end=75)
        self.assertEqual(goal_tag.kickoff_window([tag], goal, 12, 120), (50, 75))
        self.assertIsNone(goal_tag.kickoff_window([dict(tag, type='Kick Off (Start)')], goal, 12, 120))
        self.assertIsNone(goal_tag.kickoff_window([dict(tag, start=90, end=100)], goal, 12, 120))


if __name__ == '__main__':
    unittest.main()
