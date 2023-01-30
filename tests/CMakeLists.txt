find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
	add_test(
		NAME "h5bench-sync"
		COMMAND Python3::Interpreter -m pytest --verbose --rootdir ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_h5bench.py
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