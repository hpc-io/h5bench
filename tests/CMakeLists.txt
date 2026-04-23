find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
	add_test(
		NAME "h5bench-sync"
		COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_h5bench.py
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)

	# Output-correctness tests (Wave C-prep). These verify HDF5 group/
	# dataset/shape/dtype + deterministic id patterns in the files the
	# benchmarks produce. Require h5py (see tests/requirements.txt).
	add_test(
		NAME "h5bench-output-validation"
		COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_output_validation.py
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)

	# Read-path round-trip test — until now h5bench_read had no pytest coverage.
	add_test(
		NAME "h5bench-read-roundtrip"
		COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_read_roundtrip.py
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)

	# Variable-distribution write test — replaces the broken
	# test_sync_h5bench_normal_dist.py (wrong binary name, wrong sample name).
	add_test(
		NAME "h5bench-write-var-normal-dist"
		COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_write_var_normal_dist.py
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)

	# Pure-Python driver unit tests — no benchmark binary required.
	add_test(
		NAME "h5bench-driver-unit"
		COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_driver_unit.py
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)

    if(WITH_ASYNC_VOL)
		add_test(
			NAME "h5bench-async"
			COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_async_h5bench.py
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		)
	endif()

	add_test(
		NAME "h5bench-sync-metadata"
		COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_metadata.py
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)

	if(H5BENCH_EXERCISER)
		add_test(
			NAME "h5bench-sync-exerciser"
			COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_exerciser.py
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		)
	endif()

	if(H5BENCH_AMREX)
		add_test(
			NAME "h5bench-sync-amrex"
			COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_amrex.py
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		)

	    if(WITH_ASYNC_VOL)
			add_test(
				NAME "h5bench-async-amrex"
				COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_async_amrex.py
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
			)
		endif()
	endif()

	if(H5BENCH_OPENPMD)
		add_test(
			NAME "h5bench-sync-openpmd"
			COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_openpmd.py
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		)
	endif()
endif()