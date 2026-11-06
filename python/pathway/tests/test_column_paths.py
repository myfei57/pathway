from pathway.internals import column, dtype as dt
from pathway.internals.column_path import ColumnPath
from pathway.internals.column_properties import ColumnProperties
from pathway.internals.graph_runner.path_evaluator import maybe_flatten_input_storage
from pathway.internals.graph_runner.path_storage import (
    ColumnVisibility,
    ColumnVisibilityKind,
    Storage,
)


def assert_storages_have_same_paths(a: Storage, b: Storage) -> None:
    assert set(a.get_columns()) == set(b.get_columns())
    for col in a.get_columns():
        assert a.get_path(col) == b.get_path(col)


def test_removal_of_potentially_big_columns_from_storage_1():
    universe = column.Universe()
    str_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    int_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    input_storage = Storage.new(
        universe,
        {
            str_column: ColumnPath((0,)),
            int_column: ColumnPath((1,)),
        },
    )
    maybe_flat_storage = maybe_flatten_input_storage(input_storage, [int_column])
    expected_storage = Storage.new(
        universe, {int_column: ColumnPath((0,))}, is_flat=True
    )
    assert_storages_have_same_paths(maybe_flat_storage, expected_storage)


def test_removal_of_potentially_big_columns_from_storage_2():
    universe = column.Universe()
    int_column_1: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    int_column_2: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    int_column_3: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    input_storage = Storage.new(
        universe,
        {
            int_column_1: ColumnPath((0,)),
            int_column_2: ColumnPath((1,)),
            int_column_3: ColumnPath((2,)),
        },
    )
    maybe_flat_storage = maybe_flatten_input_storage(input_storage, [int_column_1])
    expected_storage = Storage.new(
        universe, {int_column_1: ColumnPath((0,))}, is_flat=True
    )
    assert_storages_have_same_paths(maybe_flat_storage, expected_storage)


def test_removal_of_potentially_big_columns_from_storage_3():
    universe = column.Universe()
    int_column_1: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    int_column_2: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    input_storage = Storage.new(
        universe,
        {
            int_column_1: ColumnPath((0,)),
            int_column_2: ColumnPath((1,)),
        },
    )
    maybe_flat_storage = maybe_flatten_input_storage(input_storage, [int_column_1])
    expected_storage = Storage.new(
        universe,
        {
            int_column_1: ColumnPath((0,)),
            int_column_2: ColumnPath((1,)),
        },
    )
    assert_storages_have_same_paths(maybe_flat_storage, expected_storage)


def test_removal_of_potentially_big_columns_from_storage_4():
    universe = column.Universe()
    str_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    int_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    input_storage = Storage.new(
        universe,
        {
            str_column: ColumnPath((0,)),
            int_column: ColumnPath((1,)),
        },
    ).restrict_to([int_column])
    # make sure also works on restricted version
    # str_column is still in the tuple structure
    maybe_flat_storage = maybe_flatten_input_storage(input_storage, [int_column])
    expected_storage = Storage.new(
        universe, {int_column: ColumnPath((0,))}, is_flat=True
    )
    assert_storages_have_same_paths(maybe_flat_storage, expected_storage)


def test_removal_of_potentially_big_columns_from_storage_5():
    universe = column.Universe()
    str_column_1: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    str_column_2: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    int_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    input_storage = Storage.new(
        universe,
        {
            str_column_1: ColumnPath((0,)),
            str_column_2: ColumnPath((1,)),
            int_column: ColumnPath((2,)),
        },
    )
    maybe_flat_storage = maybe_flatten_input_storage(
        input_storage, [int_column, str_column_2]
    )
    expected_storage = Storage.new(
        universe,
        {int_column: ColumnPath((0,)), str_column_2: ColumnPath((1,))},
        is_flat=True,
    )
    assert_storages_have_same_paths(maybe_flat_storage, expected_storage)


def test_removal_of_potentially_big_columns_from_storage_6():
    universe = column.Universe()
    str_column_1: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    str_column_2: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    int_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )
    input_storage = Storage.new(
        universe,
        {
            str_column_1: ColumnPath((0,)),
            str_column_2: ColumnPath((0,)),
            int_column: ColumnPath((2,)),
        },
    )
    # two columns with the same path one can be a reference to the other
    # expect no flatten as all fields are kept
    maybe_flat_storage = maybe_flatten_input_storage(
        input_storage, [int_column, str_column_2]
    )
    expected_storage = input_storage
    assert_storages_have_same_paths(maybe_flat_storage, expected_storage)


def _materialized_column(universe: column.Universe) -> column.Column:
    return column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.INT)
    )


def test_columns_visibility_of_visible_columns_and_keys():
    universe = column.Universe()
    other_universe = column.Universe()
    int_column = _materialized_column(universe)
    str_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    absent_column = _materialized_column(universe)
    id_column = column.MaterializedContext(universe).id_column
    other_id_column = column.MaterializedContext(other_universe).id_column

    storage = Storage.new(
        universe,
        {
            int_column: ColumnPath((0,)),
            str_column: ColumnPath((1,)),
        },
    )

    assert storage.check_columns_visibility(
        [int_column, str_column, id_column, absent_column, other_id_column]
    ) == [
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((0,))),
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((1,))),
        ColumnVisibility(ColumnVisibilityKind.KEY, ColumnPath.KEY),
        ColumnVisibility(ColumnVisibilityKind.ABSENT, None),
        ColumnVisibility(ColumnVisibilityKind.ABSENT, None),
    ]
    # the check is consistent with get_path / has_column
    assert storage.has_column(id_column)
    assert storage.get_path(id_column) == ColumnPath.KEY
    assert not storage.has_column(other_id_column)
    assert not storage.has_column(absent_column)


def test_columns_visibility_of_empty_input():
    universe = column.Universe()
    storage = Storage.new(universe, {_materialized_column(universe): ColumnPath((0,))})
    assert storage.check_columns_visibility([]) == []


def test_columns_visibility_before_and_after_restrict_to():
    universe = column.Universe()
    int_column = _materialized_column(universe)
    str_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    id_column = column.MaterializedContext(universe).id_column
    storage = Storage.new(
        universe,
        {
            int_column: ColumnPath((0,)),
            str_column: ColumnPath((1,)),
        },
    )

    # before restrict_to both columns are visible and the ID column is KEY
    assert storage.check_columns_visibility([int_column, str_column, id_column]) == [
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((0,))),
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((1,))),
        ColumnVisibility(ColumnVisibilityKind.KEY, ColumnPath.KEY),
    ]

    restricted = storage.restrict_to([int_column])

    # int_column stays visible, str_column is hidden but still in the tuple
    assert restricted.check_columns_visibility([int_column, str_column, id_column]) == [
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((0,))),
        ColumnVisibility(ColumnVisibilityKind.HIDDEN, None),
        ColumnVisibility(ColumnVisibilityKind.KEY, ColumnPath.KEY),
    ]
    assert not restricted.has_column(str_column)
    assert str_column in dict(restricted.get_all_columns()).values()

    # restrict_to doesn't modify the original storage
    assert storage.check_columns_visibility([str_column]) == [
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((1,)))
    ]


def test_columns_visibility_keeps_duplicates_in_order():
    universe = column.Universe()
    int_column = _materialized_column(universe)
    str_column = _materialized_column(universe)
    id_column = column.MaterializedContext(universe).id_column
    storage = Storage.new(
        universe,
        {
            int_column: ColumnPath((0,)),
            str_column: ColumnPath((1,)),
        },
    )

    queried = [str_column, int_column, str_column, id_column, str_column]
    assert storage.check_columns_visibility(queried) == [
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((1,))),
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((0,))),
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((1,))),
        ColumnVisibility(ColumnVisibilityKind.KEY, ColumnPath.KEY),
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((1,))),
    ]


def test_columns_visibility_of_hidden_column_from_other_universe():
    left_universe = column.Universe()
    right_universe = column.Universe()
    join_universe = column.Universe()
    left_column = _materialized_column(left_universe)
    right_column = _materialized_column(right_universe)
    join_id_column = column.MaterializedContext(join_universe).id_column
    right_id_column = column.MaterializedContext(right_universe).id_column

    merged = Storage.merge_storages(
        join_universe,
        Storage.new(left_universe, {left_column: ColumnPath((0,))}),
        Storage.new(right_universe, {right_column: ColumnPath((0,))}),
    )
    restricted = merged.restrict_to([left_column])

    assert restricted.check_columns_visibility(
        [left_column, right_column, join_id_column, right_id_column]
    ) == [
        ColumnVisibility(ColumnVisibilityKind.VISIBLE, ColumnPath((0, 0))),
        # column from the other universe still physically present in the tuple
        ColumnVisibility(ColumnVisibilityKind.HIDDEN, None),
        ColumnVisibility(ColumnVisibilityKind.KEY, ColumnPath.KEY),
        # ID column of a different universe is not a key of this storage
        ColumnVisibility(ColumnVisibilityKind.ABSENT, None),
    ]


def test_columns_visibility_is_read_only():
    universe = column.Universe()
    int_column = _materialized_column(universe)
    str_column: column.Column = column.MaterializedColumn(
        universe=universe, properties=ColumnProperties(dtype=dt.STR)
    )
    absent_column = _materialized_column(universe)
    id_column = column.MaterializedContext(universe).id_column
    other_id_column = column.MaterializedContext(column.Universe()).id_column
    storage = Storage.new(
        universe,
        {
            int_column: ColumnPath((0,)),
            str_column: ColumnPath((1,)),
        },
    ).restrict_to([int_column])

    visible_paths_before = {
        column_: storage.get_path(column_) for column_ in storage.get_columns()
    }
    all_columns_before = dict(storage.get_all_columns())

    storage.check_columns_visibility(
        [int_column, str_column, absent_column, id_column, other_id_column]
    )

    visible_paths_after = {
        column_: storage.get_path(column_) for column_ in storage.get_columns()
    }
    assert visible_paths_after == visible_paths_before
    assert dict(storage.get_all_columns()) == all_columns_before
