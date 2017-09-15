#include "catch.hpp"

#define N_RAD 2
#define POP_WIDTH 100

#include "../util/neighbor.cu"

TEST_CASE( "neighbors are calculated correctly (POP_WIDTH 100)", "[neighbor]" ) {
  
  SECTION( "normal point: 23, 78 (somewhere in the middle)" ) {
    REQUIRE( get_neighbor(23, 78, 0) == 7697 );
    REQUIRE( get_neighbor(23, 78, 1) == 7698 );
    REQUIRE( get_neighbor(23, 78, 2) == 7699 );
    REQUIRE( get_neighbor(23, 78, 3) == 7700 );
    REQUIRE( get_neighbor(23, 78, 4) == 7701 );
    REQUIRE( get_neighbor(23, 78, 5) == 7798 );
    REQUIRE( get_neighbor(23, 78, 6) == 7799 );
    REQUIRE( get_neighbor(23, 78, 7) == 7800 );
    REQUIRE( get_neighbor(23, 78, 8) == 7801 );
    REQUIRE( get_neighbor(23, 78, 9) == 7802 );
    REQUIRE( get_neighbor(23, 78, 10) == 7899 );
    REQUIRE( get_neighbor(23, 78, 11) == 7900 );
    REQUIRE( get_neighbor(23, 78, 12) == 7901 );
    REQUIRE( get_neighbor(23, 78, 13) == 7902 );
    REQUIRE( get_neighbor(23, 78, 14) == 7903 );
    REQUIRE( get_neighbor(23, 78, 15) == 8000 );
    REQUIRE( get_neighbor(23, 78, 16) == 8001 );
    REQUIRE( get_neighbor(23, 78, 17) == 8002 );
    REQUIRE( get_neighbor(23, 78, 18) == 8003 );
    REQUIRE( get_neighbor(23, 78, 19) == 8004 );
    REQUIRE( get_neighbor(23, 78, 20) == 8101 );
    REQUIRE( get_neighbor(23, 78, 21) == 8102 );
    REQUIRE( get_neighbor(23, 78, 22) == 8103 );
    REQUIRE( get_neighbor(23, 78, 23) == 8104 );
    REQUIRE( get_neighbor(23, 78, 24) == 8105 );
  }
  SECTION( "edge case: 0, 0" ) {
    REQUIRE( get_neighbor(0, 0, 0) == 9997 );
    REQUIRE( get_neighbor(0, 0, 4) == 9900 );
    REQUIRE( get_neighbor(0, 0, 5) == 10098 );
    REQUIRE( get_neighbor(0, 0, 9) == 10001 );
    REQUIRE( get_neighbor(0, 0, 10) == 99 );
    REQUIRE( get_neighbor(0, 0, 12) == 0 );
    REQUIRE( get_neighbor(0, 0, 14) == 2 );
    REQUIRE( get_neighbor(0, 0, 15) == 200 );
    REQUIRE( get_neighbor(0, 0, 19) == 103 );
    REQUIRE( get_neighbor(0, 0, 20) == 301 );
    REQUIRE( get_neighbor(0, 0, 24) == 204 );
  }
  SECTION( "edge case: 100, 0" ) {
    REQUIRE( get_neighbor(100, 0, 0) == 9996 );
    REQUIRE( get_neighbor(100, 0, 4) == 9899 );
    REQUIRE( get_neighbor(100, 0, 5) == 10097 );
    REQUIRE( get_neighbor(100, 0, 9) == 10000 );
    REQUIRE( get_neighbor(100, 0, 10) == 98 );
    REQUIRE( get_neighbor(100, 0, 12) == 100 );
    REQUIRE( get_neighbor(100, 0, 14) == 1 );
    REQUIRE( get_neighbor(100, 0, 15) == 199 );
    REQUIRE( get_neighbor(100, 0, 19) == 102 );
    REQUIRE( get_neighbor(100, 0, 20) == 300 );
    REQUIRE( get_neighbor(100, 0, 24) == 203 );
  }
  SECTION( "edge case: 0, 99" ) {
    REQUIRE( get_neighbor(0, 99, 0) == 9896 );
    REQUIRE( get_neighbor(0, 99, 4) == 9799 );
    REQUIRE( get_neighbor(0, 99, 5) == 9997 );
    REQUIRE( get_neighbor(0, 99, 9) == 9900 );
    REQUIRE( get_neighbor(0, 99, 10) == 10098 );
    REQUIRE( get_neighbor(0, 99, 12) == 9999 );
    REQUIRE( get_neighbor(0, 99, 14) == 10001 );
    REQUIRE( get_neighbor(0, 99, 15) == 99 );
    REQUIRE( get_neighbor(0, 99, 19) == 2 );
    REQUIRE( get_neighbor(0, 99, 20) == 200 );
    REQUIRE( get_neighbor(0, 99, 24) == 103 );
  }
  SECTION( "edge case: 100, 99" ) {
    REQUIRE( get_neighbor(100, 99, 0) == 9895 );
    REQUIRE( get_neighbor(100, 99, 4) == 9798 );
    REQUIRE( get_neighbor(100, 99, 5) == 9996 );
    REQUIRE( get_neighbor(100, 99, 9) == 9899 );
    REQUIRE( get_neighbor(100, 99, 10) == 10097 );
    REQUIRE( get_neighbor(100, 99, 12) == 10099 );
    REQUIRE( get_neighbor(100, 99, 14) == 10000 );
    REQUIRE( get_neighbor(100, 99, 15) == 98 );
    REQUIRE( get_neighbor(100, 99, 19) == 1 );
    REQUIRE( get_neighbor(100, 99, 20) == 199 );
    REQUIRE( get_neighbor(100, 99, 24) == 102 );
  }
}

