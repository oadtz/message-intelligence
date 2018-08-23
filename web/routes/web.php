<?php


Route::prefix('api')->group(function () {
    Route::get('flights', 'Api\FlightController@getFlights');
    Route::post('flights', 'Api\FlightController@addFlight');
    Route::any('chatbot', 'Api\ChatbotController@chat');
});

Route::get('/flights', 'FlightController@index');
Route::get('/chatbot', 'SiteController@chatbot');
Route::get('/', 'SiteController@index');
