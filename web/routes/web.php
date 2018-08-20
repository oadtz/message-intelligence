<?php


Route::prefix('api')->group(function () {
    Route::get('flights', 'ApiFlightController@getFlights');
    Route::post('flights', 'ApiFlightController@addFlight');
});

Route::get('/', function () {
    return view('welcome');
});
