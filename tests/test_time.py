# Copyright (c) 2022 The Aerospace Corporation

import numpy as np
import pytest
import copy
import datetime
import gps_time as time

from gps_time import GPSTime
from gps_time.datetime import (
    tow2datetime,
    datetime2tow,
    tow2zcount,
    zcount2tow,
    datetime2zcount,
    zcount2datetime,
    arange_datetime,
    correct_week,
)
from gps_time.utilities import arange_gpstime, validate_gps_week

from gps_frames.parameters import GPSparam

SEC_IN_WEEK = 7 * 24 * 3600

# Detect if gps_time returns timezone-aware datetimes
try:
    _test_dt = tow2datetime(0, 0)
    HAS_TZ = _test_dt.tzinfo is not None
except Exception:
    HAS_TZ = False

EXPECTED_TZ = datetime.timezone.utc if HAS_TZ else None


@pytest.mark.parametrize("week, sec", [(0, 0), (2000, -3), (2032, 604801)])
def test_init(week, sec):

    _time = GPSTime(week, sec)

    assert _time.time_of_week < SEC_IN_WEEK
    assert _time.time_of_week >= 0


@pytest.mark.parametrize("week, sec", [(0, 0), (2000, -3), (2032, 604801)])
def test_datetime(week, sec):

    _time = GPSTime(week, sec)

    dt = tow2datetime(_time.week_number, _time.time_of_week)
    assert _time.to_datetime() == dt

    try:
        GPSTime.from_datetime("foo")
        assert -1 == 1
    except TypeError:
        pass

    assert GPSTime.from_datetime(dt) == _time


@pytest.mark.parametrize("week, sec", [(0, 0), (2000, -3), (2032, 604801)])
def test_zcount(week, sec):

    _time = GPSTime(week, sec)

    assert _time.to_zcount() == _time.time_of_week / 1.5


def test_correct_weeks():

    _time = GPSTime(2600, 0)
    _time.time_of_week -= 10
    _time.correct_weeks()

    assert _time.week_number == 2599

    _time.time_of_week += 10 + SEC_IN_WEEK
    _time.correct_weeks()
    assert _time.week_number == 2601


def test_add():

    _time = GPSTime(2600, 0)

    _time = _time.__add__(1)
    assert _time.time_of_week == 1
    _time = _time.__add__(-2.0)
    assert _time.time_of_week == SEC_IN_WEEK - 1

    td = datetime.timedelta(0, 1, 0, 0, 0, 6, 3)
    assert _time.__add__(td).week_number == _time.week_number + 4
    assert _time.__add__(td).time_of_week == (_time.time_of_week + 21601) % SEC_IN_WEEK

    _time2 = GPSTime(3, -10)
    dt = _time2.to_datetime()

    assert _time.__add__(dt).week_number == _time.week_number + 3
    assert _time.__add__(dt).time_of_week == _time.time_of_week - 10

    assert _time.__add__(_time2).week_number == _time.week_number + 3
    assert _time.__add__(_time2).time_of_week == _time.time_of_week - 10

    times = _time.__add__(np.array([_time2, _time2]))

    assert times[0] == _time.__add__(_time2)
    assert times[1] == _time.__add__(_time2)

    try:
        _time.__add__("foo")
        assert -1 == 1
    except TypeError:
        pass


def test_sub():

    _time = GPSTime(2600, 0)

    assert _time.__sub__(5.0).time_of_week == SEC_IN_WEEK - 5
    assert _time.__sub__(5).week_number == 2599

    td = datetime.timedelta(0, 0, 0, 0, 0, 6, 3)

    assert _time.__sub__(td).week_number == _time.week_number - 4
    assert (
        _time.__sub__(td).time_of_week == (_time.time_of_week - 6 * 3600) % SEC_IN_WEEK
    )

    _time2 = GPSTime(2594, SEC_IN_WEEK - 5)
    dt = _time2.to_datetime()

    assert _time.__sub__(dt) == 5 * SEC_IN_WEEK + 5
    assert _time.__sub__(_time2) == 5 * SEC_IN_WEEK + 5
    nums = np.array([5, 3, 4])
    times = _time.__sub__(nums)

    assert times[0] == _time.__sub__(5)
    assert times[1] == _time.__sub__(3)
    assert times[2] == _time.__sub__(4)

    # Function seems to  not be handling array of datetime objects properly
    # not returning datetime and datetime.timedelta class names when calling
    # datetime.datetime.__class__.__name__ function and datetime.time.delta.__class.__name, respectively
    # returns "type" instead
    dts = np.array([dt, dt])
    times2 = _time.__sub__(dts)
    assert times2[0] == _time.__sub__(dt)
    assert times2[1] == _time.__sub__(dt)

    tds = np.array([td, td])
    times2 = _time.__sub__(tds)
    assert times2[0] == _time.__sub__(td)
    assert times2[1] == _time.__sub__(td)

    # gpstimes
    gts = np.array([_time2, _time2])
    times2 = _time.__sub__(gts)
    assert times2[0] == _time.__sub__(_time2)
    assert times2[1] == _time.__sub__(_time2)

    try:
        _time.__sub__("foo")
        assert -1 == 1
    except TypeError:
        pass


@pytest.mark.parametrize(
    "time1, time2",
    [
        (GPSTime(245, 5782), GPSTime(245, 5783)),
        (GPSTime(245, 5782), GPSTime(689, 5783)),
    ],
)
def test_lt(time1, time2):

    assert time1.__lt__(time2)
    assert not time2.__lt__(time1)

    assert time1.__lt__(time2.to_datetime())
    assert not time2.__lt__(time1.to_datetime())

    try:
        time1.__lt__("foo")
        assert -1 == 1
    except TypeError:
        pass


@pytest.mark.parametrize(
    "time1, time2",
    [
        (GPSTime(245, 5782), GPSTime(245, 5783)),
        (GPSTime(245, 5782), GPSTime(689, 5783)),
    ],
)
def test_gt(time1, time2):

    assert time2.__gt__(time1)
    assert not time1.__gt__(time2)

    assert time2.__gt__(time1.to_datetime())
    assert not time1.__gt__(time2.to_datetime())

    try:
        time1.__gt__("foo")
        assert -1 == 1
    except TypeError:
        pass


@pytest.mark.parametrize(
    "time1, time2",
    [
        (GPSTime(245, 5782), GPSTime(245, 5783)),
        (GPSTime(245, 5782), GPSTime(689, 5783)),
    ],
)
def test_eq(time1, time2):

    assert time1.__eq__(time1)
    assert not time1.__eq__(time2)

    assert time1.__eq__(time1.to_datetime())
    assert not time1.__eq__(time2.to_datetime())

    try:
        time1.__eq__("foo")
        assert -1 == 1
    except TypeError:
        pass


@pytest.mark.parametrize(
    "time1, time2",
    [
        (GPSTime(245, 5782), GPSTime(245, 5783)),
        (GPSTime(245, 5782), GPSTime(689, 5783)),
    ],
)
def test_le(time1, time2):

    assert time1.__le__(time2)
    assert time1.__le__(time1)
    assert not time2.__le__(time1)

    assert time1.__le__(time2.to_datetime())
    assert time1.__le__(time1.to_datetime())
    assert not time2.__le__(time1.to_datetime())

    try:
        time1.__le__("foo")
        assert -1 == 1
    except TypeError:
        pass


@pytest.mark.parametrize(
    "time1, time2",
    [
        (GPSTime(245, 5782), GPSTime(245, 5783)),
        (GPSTime(245, 5782), GPSTime(689, 5783)),
    ],
)
def test_ge(time1, time2):

    assert time2.__ge__(time1)
    assert time2.__ge__(time2)
    assert not time1.__ge__(time2)

    assert time2.__ge__(time1.to_datetime())
    assert time2.__ge__(time2.to_datetime())
    assert not time1.__ge__(time2.to_datetime())

    try:
        time1.__ge__("foo")
        assert -1 == 1
    except TypeError:
        pass


@pytest.mark.parametrize(
    "time1, time2",
    [
        (GPSTime(245, 5782), GPSTime(245, 5783)),
        (GPSTime(245, 5782), GPSTime(689, 5783)),
    ],
)
def test_ne(time1, time2):

    assert not time1.__ne__(time1)
    assert time1.__ne__(time2)

    assert not time1.__ne__(time1.to_datetime())
    assert time1.__ne__(time2.to_datetime())

    try:
        time1.__ne__("foo")
        assert -1 == 1
    except TypeError:
        pass


@pytest.mark.parametrize("weeks, secs", [(245, 48761), (2, SEC_IN_WEEK - 3)])
def test_iadd(weeks, secs):

    time1 = GPSTime(weeks, secs)
    time2 = GPSTime(42, 6231)

    time3 = copy.copy(time1)

    time1 += time2
    assert time1 == time3 + time2

    time3 += time2.to_datetime()
    assert time1 == time3

    td = datetime.timedelta(3, 12, 4, 5, 7, 5123, 6)
    time1 += td
    assert time1 == time3 + td


@pytest.mark.parametrize("weeks, secs", [(245, 48761), (2, SEC_IN_WEEK - 3)])
def test_isub(weeks, secs):

    time1 = GPSTime(weeks, secs)
    time2 = GPSTime(42, 6231)

    time3 = copy.copy(time1)

    # try:
    #     time1 -= time2
    #     assert(-1 == 1)
    # except TypeError:
    #     pass

    delt = SEC_IN_WEEK - 45
    time1 -= delt
    assert time1 == time3 - delt
    td = datetime.timedelta(0, delt, 0, 0, 0, 0, 0)
    time3 -= td
    assert time3 == time1


@pytest.mark.parametrize(
    "time1",
    [
        (GPSTime(245, 5782)),
        (GPSTime(245, 5783)),
        (GPSTime(245, 5782)),
        (GPSTime(689, 5783)),
    ],
)
def test_rep(time1):

    assert str(time1) == "GPSTime(week_number={}, time_of_week={})".format(
        time1.week_number, time1.time_of_week
    )


# def test_iso_datetime():
#     time1 = GPSTime(2111, 5 * 86400 + 18 * 3600 + 17.34)

#     dt = time1.to_datetime()
#     assert(time.cast_to_datetime('2020-06-26T18:00:17.340000') == dt)

#     time1 = GPSTime(2111, 5 * 86400 + 18 * 3600 + 17)

#     dt = time1.to_datetime()
#     assert(time.cast_to_datetime('2020-06-26T18:00:17') == dt)

#     try:
#         time.cast_to_datetime('foo')
#         assert('foo' == 'ISO')
#     except IOError:
#         pass

#     # reverse reverse
#     assert(time.datetime_to_iso(dt) == '2020-06-26T18:00:17')


def test_validate_gps():

    validate_gps_week(2024, 1000)

    try:
        validate_gps_week(2024, 1001)
        assert -1 == 1
    except ValueError:
        pass


@pytest.mark.parametrize("delta", [80.0, 17.6, 132.9])
def test_arange_gpstime(delta):
    _time = GPSTime(1923, SEC_IN_WEEK - 1082.0)
    truth_time = GPSTime(1923, SEC_IN_WEEK - 1082.0) + np.arange(
        0.0, 3600.0, delta / 1000.0
    )

    times = arange_gpstime(_time, 3600, delta)

    errors = times - truth_time

    print("*" * 50, np.max(np.abs(errors)))

    assert np.all(np.abs(errors) < 1.0e-12)


@pytest.mark.parametrize("delta", [80.0, 17.6, 132.9])
def test_arange_datetime(delta):

    _time = GPSTime(1923, SEC_IN_WEEK - 1082)
    dtime = _time.to_datetime()

    times = arange_datetime(dtime, 3600, delta)

    for t in times:
        assert dtime == t
        dtime += datetime.timedelta(0, 0, 0, delta, 0, 0, 0)


@pytest.mark.parametrize("weeks", [0, 12, 342, 1367, 2012])
def test_correct_week(weeks):

    expected = weeks
    tow = 0.0

    year = (
        GPSparam.epoch_datetime + datetime.timedelta(days=7 * weeks, seconds=tow)
    ).year

    weeks = weeks % 1024

    try:
        correct_week(weeks, tow, float(year))
        assert "float" == "int"
    except ValueError:
        pass

    assert correct_week(weeks, tow, year) == expected

    try:
        correct_week(weeks, tow, year + 1)
        assert -1 == 1
    except ValueError:
        pass


@pytest.mark.parametrize("weeks, tow", [(52, 0), (243, 209312), (2182, 9982)])
def test_tow2datetime(weeks, tow):

    dt = datetime.datetime(1980, 1, 6, 0, 0, 0, 0, tzinfo=EXPECTED_TZ)

    dt += datetime.timedelta(days=weeks * 7)
    dt += datetime.timedelta(seconds=tow)

    assert dt == tow2datetime(weeks, tow)
    assert dt == tow2datetime(weeks, tow)

    w, t = datetime2tow(dt)
    assert w == weeks
    assert t == tow

    try:
        datetime2tow("foo")
        assert "foo" == "datetime"
    except TypeError:
        pass
    dt = datetime.datetime(1980, 1, 6, 0, 0, 0, 0, tzinfo=EXPECTED_TZ)
    assert tow2datetime(52, 0, 1981) == dt + datetime.timedelta(days=7 * 52)


# def test_array_time_diff():

#     dt1 = datetime.datetime(2003, 10, 7, 16, 37, 59, 123)
#     dt2 = datetime.datetime(1987, 3, 22, 23, 13, 46, 11)
#     dt3 = datetime.datetime(2014, 6, 1, 19, 53, 26, 98)
#     dt4 = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)

#     dts1 = [dt1, dt2]
#     dts2 = [dt3, dt4]

#     try:
#         time.array_time_difference(dts2, dts1)
#         assert('datetimearray' == 'numpyarray')
#     except TypeError:
#         pass

#     try:
#         time.array_time_difference('foo', dts1)
#         assert('str' == 'dt')
#     except TypeError:
#         pass

#     try:
#         time.array_time_difference(np.array([1]), dt2)
#         assert(-1 == 1)
#     except TypeError:
#         pass

#     try:
#         time.array_time_difference(dt3, np.array([1]))
#         assert(-1 == 1)
#     except TypeError:
#         pass

#     dts1 = np.array(dts1)
#     dts2 = np.array(dts2)

#     out = time.array_time_difference(dts2, dts1)

#     for i in range(len(out)):
#         td = dts2[i] - dts1[i]
#         assert(out[i] == td.total_seconds())

#     out = time.array_time_difference(dt1, dt2)
#     assert(out[0] == (dt1 - dt2).total_seconds())


# def test_diff_seconds():

#     dt1 = datetime.datetime(2003, 10, 7, 16, 37, 59, 123)
#     dt2 = datetime.datetime(1987, 3, 22, 23, 13, 46, 11)
#     dt3 = datetime.datetime(2014, 6, 1, 19, 53, 26, 98)
#     dt4 = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)

#     dts = [dt1, dt2, dt3]
#     out = time.diff_seconds(dt4, dts)

#     for i in range(len(out)):
#         assert(out[i] == (dt4 - dts[i]).total_seconds())


# def test_sub_timedelt():

#     dt1 = datetime.datetime(2003, 10, 7, 16, 37, 59, 123)
#     dt2 = datetime.datetime(1987, 3, 22, 23, 13, 46, 11)
#     dt3 = datetime.datetime(2014, 6, 1, 19, 53, 26, 98)
#     dt4 = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)

#     dts = [dt1, dt2, dt3, dt4]

#     tds = [12, 78, 8712, SEC_IN_WEEK - 19]

#     out = time.subtract_timedelta(dts, tds)

#     for i in range(len(out)):
#         assert(out[i] == dts[i] - datetime.timedelta(seconds=tds[i]))


# def test_sub_timedelt_tow():

#     dt1 = datetime.datetime(2003, 10, 7, 16, 37, 59, 123)
#     dt2 = datetime.datetime(1987, 3, 22, 23, 13, 46, 11)
#     dt3 = datetime.datetime(2014, 6, 1, 19, 53, 26, 98)
#     dt4 = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)

#     dts = [dt1, dt2, dt3, dt4]

#     tds = [12, 78, 8712, SEC_IN_WEEK - 19]

#     out = time.subtract_timedelta_as_tow(dts, tds)

#     for i in range(len(out)):

#         tow = datetime2tow(dts[i] - datetime.timedelta(seconds=tds[i]))
#         assert((out[i] == tow).all())


# def test_get_tow():

#     try:
#         time.get_time_of_week()
#     except NotImplementedError:
#         pass


@pytest.mark.parametrize("weeks", [0.0, 12.0, 342.0, 1367.0, 2012.0])
def test_tow2zcount(weeks):

    expected = weeks
    tow = 235

    year = (
        GPSparam.epoch_datetime + datetime.timedelta(days=7 * weeks, seconds=tow)
    ).year

    assert tow2zcount(weeks, tow) == (weeks, tow / 1.5)
    assert tow2zcount(weeks, tow, year) == (weeks, tow / 1.5)

    tow = SEC_IN_WEEK - 38

    year = (
        GPSparam.epoch_datetime + datetime.timedelta(days=7 * weeks, seconds=tow)
    ).year

    assert tow2zcount(weeks, tow) == (weeks, tow / 1.5)
    assert tow2zcount(weeks, tow, year) == (weeks, tow / 1.5)

    z = tow2zcount(weeks, tow)
    assert zcount2tow(z[0], z[1]) == (weeks, tow)
    assert zcount2tow(z[0], z[1], year) == (weeks, tow)


@pytest.mark.parametrize(
    "dt",
    [
        datetime.datetime(2003, 10, 7, 16, 37, 59, 123, tzinfo=EXPECTED_TZ),
        datetime.datetime(1987, 3, 22, 23, 13, 46, 11, tzinfo=EXPECTED_TZ),
        datetime.datetime(2014, 6, 1, 19, 53, 26, 98, tzinfo=EXPECTED_TZ),
        datetime.datetime(2020, 1, 1, 0, 0, 0, 0, tzinfo=EXPECTED_TZ),
    ],
)
def test_datetime_tozcount(dt):

    weeks, zcount = datetime2zcount(dt)

    expctWeeks, tow = datetime2tow(dt)

    assert weeks == expctWeeks
    assert zcount == tow / 1.5

    assert zcount2datetime(weeks, zcount) == dt
